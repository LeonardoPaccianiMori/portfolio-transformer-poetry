"""Standard-library web server for selected-adapter sonnet generation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

from sonnet_evaluation.minerva_7b_sonnet_candidates import (
    build_sonnet_candidate_prompt,
)
from sonnet_evaluation.minerva_7b_sonnet_final import validate_frozen_selection
from sonnet_evaluation.minerva_generation import (
    _load_dependencies,
    generate_minerva_continuation,
)
from sonnet_training.cuda_compat import prepare_cuda_memory_measurement
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


DEMO_VERSION = "selected_minerva_7b_v6_demo_v1"
DEMO_MAX_NEW_TOKENS = 512
DEMO_TOP_K = 50
DEMO_CONTINUATION_LINES = 13
DEFAULT_TEMPERATURE = 0.8
DEFAULT_SEED = 1337
MIN_TEMPERATURE = 0.5
MAX_TEMPERATURE = 1.2
MAX_OPENING_CHARACTERS = 240


@dataclass(frozen=True)
class DemoGenerationRequest:
    opening_line: str
    temperature: float = DEFAULT_TEMPERATURE
    seed: int = DEFAULT_SEED


def parse_generation_request(payload: Any) -> DemoGenerationRequest:
    """Validate one JSON request before allocating generation work."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    opening_line = payload.get("opening_line")
    if not isinstance(opening_line, str):
        raise ValueError("opening_line must be a string")
    opening_line = opening_line.strip()
    if not opening_line:
        raise ValueError("opening_line must not be empty")
    if "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one line")
    if len(opening_line) > MAX_OPENING_CHARACTERS:
        raise ValueError(
            f"opening_line must be at most {MAX_OPENING_CHARACTERS} characters"
        )

    temperature = payload.get("temperature", DEFAULT_TEMPERATURE)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be numeric")
    temperature = float(temperature)
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
        )

    seed = payload.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be between 0 and 2147483647")
    return DemoGenerationRequest(
        opening_line=opening_line,
        temperature=temperature,
        seed=seed,
    )


class SelectedSonnetGenerator:
    """Serialize GPU generation through one loaded selected adapter."""

    is_ready = True

    def __init__(self, *, model: Any, tokenizer: Any, device: torch.device | str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self._lock = threading.Lock()

    def generate(self, request: DemoGenerationRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        with self._lock:
            result = generate_minerva_continuation(
                model=self.model,
                tokenizer=self.tokenizer,
                opening_line=request.opening_line,
                max_new_tokens=DEMO_MAX_NEW_TOKENS,
                device=self.device,
                seed=request.seed,
                temperature=request.temperature,
                top_k=DEMO_TOP_K,
                continuation_line_target=DEMO_CONTINUATION_LINES,
                conditioning_prompt=build_sonnet_candidate_prompt(
                    self.tokenizer, request.opening_line
                ),
            )
        return {
            "demo_version": DEMO_VERSION,
            "request": asdict(request),
            "text": result["text"],
            "line_count": len([
                line for line in result["text"].splitlines() if line.strip()
            ]),
            "stop_reason": result["stop_reason"],
            "generated_new_tokens": result["generated_new_tokens"],
            "elapsed_seconds": time.perf_counter() - started_at,
        }


class StaticDemoGenerator:
    """Serve the interface without loading model weights for visual checks."""

    is_ready = False

    def generate(self, request: DemoGenerationRequest) -> dict[str, Any]:
        del request
        raise RuntimeError("model is not loaded in static-only mode")


def load_selected_sonnet_generator(
    *,
    repo_root: Path,
    run_dir: Path,
    selection_path: Path,
    candidate_summary_path: Path,
    cache_dir: Path,
    device: torch.device | str,
    progress: Any = None,
) -> SelectedSonnetGenerator:
    """Load the frozen epoch-4 adapter over an NF4 inference base."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the selected Minerva 7B demo requires CUDA")
    run_dir = _resolve(repo_root, run_dir)
    selection_path = _resolve(repo_root, selection_path)
    candidate_summary_path = _resolve(repo_root, candidate_summary_path)
    cache_dir = _resolve(repo_root, cache_dir)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    epoch, checkpoint_path = validate_frozen_selection(
        selection=selection,
        run_dir=run_dir,
        candidate_summary_path=candidate_summary_path,
    )
    _report(progress, f"validated frozen selected adapter epoch={epoch}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    recipe = checkpoint["recipe_config"]
    dependencies = _load_dependencies()

    _report(progress, "loading pinned Minerva 7B tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    build_sonnet_candidate_prompt(tokenizer, "Amor mi guida ancora")

    _report(progress, "loading Minerva 7B base in 4-bit NF4 for inference")
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    prepare_cuda_memory_measurement(resolved_device)
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map={"": resolved_device.index or 0},
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    _report(progress, "attaching selected rank-8 attention adapter")
    model = dependencies["get_peft_model"](
        model,
        dependencies["LoraConfig"](
            task_type="CAUSAL_LM",
            r=recipe["lora_rank"],
            lora_alpha=recipe["lora_alpha"],
            lora_dropout=recipe["lora_dropout"],
            bias="none",
            target_modules=list(recipe["target_modules"]),
        ),
    )
    dependencies["set_peft_model_state_dict"](
        model, checkpoint["adapter_state_dict"]
    )
    model.eval()
    model.config.use_cache = True
    return SelectedSonnetGenerator(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
    )


def create_demo_handler(
    *, static_root: Path, generator: SelectedSonnetGenerator | StaticDemoGenerator
) -> type[BaseHTTPRequestHandler]:
    """Bind static assets and one generator to an HTTP request handler."""
    resolved_static_root = static_root.resolve()
    asset_paths = {
        "/": (resolved_static_root / "index.html", "text/html; charset=utf-8"),
        "/assets/styles.css": (
            resolved_static_root / "styles.css",
            "text/css; charset=utf-8",
        ),
        "/assets/app.js": (
            resolved_static_root / "app.js",
            "text/javascript; charset=utf-8",
        ),
    }

    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "SonnetDemo/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/status":
                ready = generator.is_ready
                self._write_json({
                    "status": "ready" if ready else "static_only",
                    "demo_version": DEMO_VERSION,
                    "model": "Minerva 7B V6 selected epoch 4",
                })
                return
            asset = asset_paths.get(path)
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            asset_path, content_type = asset
            try:
                content = asset_path.read_bytes()
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/generate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 16_384:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(content_length))
                request = parse_generation_request(payload)
                result = generator.generate(request)
            except (ValueError, json.JSONDecodeError) as error:
                self._write_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            except RuntimeError as error:
                self._write_json(
                    {"error": str(error)},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            self._write_json(result)

        def _write_json(
            self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            content = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, message: str, *args: Any) -> None:
            print(f"demo-http | {self.address_string()} | {message % args}", flush=True)

    return DemoHandler


def serve_demo(
    *,
    host: str,
    port: int,
    static_root: Path,
    generator: SelectedSonnetGenerator | StaticDemoGenerator,
) -> None:
    """Serve the local interface until interrupted."""
    handler = create_demo_handler(static_root=static_root, generator=generator)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"demo | ready url=http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("demo | stopping", flush=True)
    finally:
        server.server_close()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Any, message: str) -> None:
    if progress is not None:
        progress(message)
