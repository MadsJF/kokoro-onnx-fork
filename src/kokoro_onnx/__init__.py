import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import time
from collections.abc import AsyncGenerator

import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray

from .config import MAX_PHONEME_LENGTH, SAMPLE_RATE, EspeakConfig, KoKoroConfig
from .log import log
from .tokenizer import Tokenizer
from .trim import trim as trim_audio


class Kokoro:
    def __init__(
        self,
        model_path: str,
        voices_path: str,
        espeak_config: EspeakConfig | None = None,
        vocab_config: dict | str | None = None,
        force_token_length: bool = False,
    ):
        # Show useful information for bug reports
        log.debug(
            f"koko-onnx version {importlib.metadata.version('kokoro-onnx')} on {platform.platform()} {platform.version()}"
        )
        self.config = KoKoroConfig(model_path, voices_path, espeak_config)
        self.config.validate()

        # See list of providers https://github.com/microsoft/onnxruntime/issues/22101#issuecomment-2357667377
        providers = ["CPUExecutionProvider"]

        # Check if kokoro-onnx installed with kokoro-onnx[gpu] feature (Windows/Linux)
        gpu_enabled = importlib.util.find_spec("onnxruntime-gpu")
        if gpu_enabled:
            providers: list[str] = rt.get_available_providers()

        # Check if ONNX_PROVIDER environment variable was set
        env_provider = os.getenv("ONNX_PROVIDER")
        if env_provider:
            providers = [env_provider]

        log.debug(f"Providers: {providers}")
        self.sess = rt.InferenceSession(model_path, providers=providers)
        self.voices: np.ndarray = np.load(voices_path)

        vocab = self._load_vocab(vocab_config)
        self.tokenizer = Tokenizer(espeak_config, vocab=vocab)

        self.force_token_length = force_token_length
        self.target_token_length = MAX_PHONEME_LENGTH-1

    @classmethod
    def from_session(
        cls,
        session: rt.InferenceSession,
        voices_path: str,
        espeak_config: EspeakConfig | None = None,
        vocab_config: dict | str | None = None,
    ):
        instance = cls.__new__(cls)
        instance.sess = session
        instance.config = KoKoroConfig(session._model_path, voices_path, espeak_config)
        instance.config.validate()
        instance.voices = np.load(voices_path)

        vocab = instance._load_vocab(vocab_config)
        instance.tokenizer = Tokenizer(espeak_config, vocab=vocab)
        return instance

    def _load_vocab(self, vocab_config: dict | str | None) -> dict:
        """Load vocabulary from config file or dictionary.

        Args:
            vocab_config: Path to vocab config file or dictionary containing vocab.

        Returns:
            Loaded vocabulary dictionary or empty dictionary if no config provided.
        """

        if isinstance(vocab_config, str):
            with open(vocab_config, encoding="utf-8") as fp:
                config = json.load(fp)
                return config["vocab"]
        if isinstance(vocab_config, dict):
            return vocab_config["vocab"]
        return {}

    def _create_audio(
        self, phonemes: str, voice: NDArray[np.float32], speed: float
    ) -> tuple[NDArray[np.float32], int]:
        log.debug(f"Phonemes: {phonemes}")
        if len(phonemes) > MAX_PHONEME_LENGTH:
            log.warning(
                f"Phonemes are too long, truncating to {MAX_PHONEME_LENGTH} phonemes"
            )
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        start_t = time.time()
        raw_tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
        assert len(raw_tokens) <= MAX_PHONEME_LENGTH, (
            f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad token 0 at the start & end"
        )

        log.debug(f"Phoneme length: {len(phonemes)}")
        log.debug(f"Raw token length: {len(raw_tokens)}")

        if self.force_token_length:
            # pad tokens up to point
            padded_tokens = [0] * self.target_token_length

            for i, token_id in enumerate(raw_tokens):
                padded_tokens[i + 1] = token_id

            voice = voice[self.target_token_length]
            tokens = [padded_tokens]
        else:
            tokens = raw_tokens
            voice = voice[len(tokens)]
            tokens = [[0, *tokens, 0]]

        log.debug(f"Token length going into model is: {(np.array(tokens, dtype=np.int64)).shape[1]}")

        if "input_ids" in [i.name for i in self.sess.get_inputs()]:
            # Newer export versions
            inputs = {
                "input_ids": tokens,
                "style": np.array(voice, dtype=np.float32),
                "speed": np.array([speed], dtype=np.int32),
            }
        else:
            inputs = {
                "tokens": tokens,
                "style": voice,
                "speed": np.ones(1, dtype=np.float32) * speed,
            }

        audio = self.sess.run(None, inputs)[0]
        audio_duration = len(audio) / SAMPLE_RATE
        create_duration = time.time() - start_t
        rtf = create_duration / audio_duration
        log.debug(
            f"Created audio in length of {audio_duration:.2f}s for {len(phonemes)} phonemes in {create_duration:.2f}s (RTF: {rtf:.2f}"
        )
        return audio, SAMPLE_RATE

    def get_voice_style(self, name: str) -> NDArray[np.float32]:
        return self.voices[name]

    def _split_phonemes(self, phonemes: str) -> list[str]:
        """
        Split phonemes into batches of MAX_PHONEME_LENGTH
        Prefer splitting at punctuation marks.
        """
        # safety net for max phonemes
        safe_ceiling = MAX_PHONEME_LENGTH - 10

        # Regular expression to split by punctuation and keep them
        words = re.split(r"([.,!?;])", phonemes)
        batched_phoenemes: list[str] = []
        current_batch = ""

        i = 0
        while i < len(words):
            part = words[i]
            # Remove leading/trailing whitespace
            part = part.strip()

            if not part:
                i += 1
                continue

            # prevent part from being over max length
            if len(part) + 1 >= MAX_PHONEME_LENGTH:
                log.debug(f"part length is: {len(part)}")
                log.debug(f"SPlitting phoneme at part: {i}")
                split_idx = part.rfind(" ", 0, safe_ceiling)

                if split_idx == -1:
                        # Emergency fallback: If there are literally NO spaces (one giant word),
                        # we are forced to hard-split at the ceiling character index
                        split_idx = safe_ceiling

                first_half = part[:split_idx].strip()
                second_half = part[split_idx:].strip()

                words[i] = first_half
                
                if second_half:
                    words.insert(i + 1, second_half)

                part = first_half
                log.debug(f"part updated to have length: {len(part)}")


            # If adding the part exceeds the max length, split into a new batch
            # TODO: make it more accurate
            if len(current_batch) + len(part) + 1 >= MAX_PHONEME_LENGTH:
                batched_phoenemes.append(current_batch.strip())
                current_batch = part
            else:
                if part in ".,!?;":
                    current_batch += part
                else:
                    if current_batch:
                        current_batch += " "
                    current_batch += part
            i += 1
        # Append the last batch if it contains any phonemes
        if current_batch:
            batched_phoenemes.append(current_batch.strip())
        return batched_phoenemes

    def create(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        trim: bool = True,
    ) -> tuple[NDArray[np.float32], int]:
        """
        Create audio from text using the specified voice and speed.
        """
        assert speed >= 0.5 and speed <= 2.0, "Speed should be between 0.5 and 2.0"

        if isinstance(voice, str):
            assert voice in self.voices, f"Voice {voice} not found in available voices"
            voice = self.get_voice_style(voice)

        start_t = time.time()
        if is_phonemes:
            phonemes = text
        else:
            phonemes = self.tokenizer.phonemize(text, lang)
        # Create batches of phonemes by splitting spaces to MAX_PHONEME_LENGTH
        batched_phoenemes = self._split_phonemes(phonemes)

        audio = []
        log.debug(
            f"Creating audio for {len(batched_phoenemes)} batches for {len(phonemes)} phonemes"
        )
        for phonemes in batched_phoenemes:
            audio_part, _ = self._create_audio(phonemes, voice, speed)
            if trim:
                # Trim leading and trailing silence for a more natural sound concatenation
                # (initial ~2s, subsequent ~0.02s)
                audio_part, _ = trim_audio(audio_part)
            audio.append(audio_part)
        audio = np.concatenate(audio)
        log.debug(f"Created audio in {time.time() - start_t:.2f}s")
        return audio, SAMPLE_RATE

    async def create_stream(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        trim: bool = True,
    ) -> AsyncGenerator[tuple[NDArray[np.float32], int], None]:
        """
        Stream audio creation asynchronously in the background, yielding chunks as they are processed.
        """
        assert speed >= 0.5 and speed <= 2.0, "Speed should be between 0.5 and 2.0"

        if isinstance(voice, str):
            assert voice in self.voices, f"Voice {voice} not found in available voices"
            voice = self.get_voice_style(voice)

        if is_phonemes:
            phonemes = text
        else:
            phonemes = self.tokenizer.phonemize(text, lang)

        batched_phonemes = self._split_phonemes(phonemes)
        queue: asyncio.Queue[tuple[NDArray[np.float32], int] | None] = asyncio.Queue()

        async def process_batches():
            """Process phoneme batches in the background."""
            for i, phonemes in enumerate(batched_phonemes):
                loop = asyncio.get_event_loop()
                # Execute in separate thread since it's blocking operation
                audio_part, sample_rate = await loop.run_in_executor(
                    None, self._create_audio, phonemes, voice, speed
                )
                if trim:
                    # Trim leading and trailing silence for a more natural sound concatenation
                    # (initial ~2s, subsequent ~0.02s)
                    audio_part, _ = trim_audio(audio_part)
                log.debug(f"Processed chunk {i} of stream")
                await queue.put((audio_part, sample_rate))
            await queue.put(None)  # Signal the end of the stream

        # Start processing in the background
        asyncio.create_task(process_batches())

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def get_voices(self) -> list[str]:
        return list(sorted(self.voices.keys()))
