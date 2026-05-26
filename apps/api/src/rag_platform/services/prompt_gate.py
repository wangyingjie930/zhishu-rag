import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROMPT_VARIABLE_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


@dataclass
class PromptGateReport:
    checked_prompts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def merge(self, other: "PromptGateReport") -> None:
        self.checked_prompts.extend(other.checked_prompts)
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checked_prompts": list(dict.fromkeys(self.checked_prompts)),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def load_prompt_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_prompt_gate(
    manifest: Dict[str, Any],
    event_payload: Optional[Dict[str, Any]] = None,
) -> PromptGateReport:
    report = validate_manifest(manifest)
    prompt_payload = extract_langfuse_prompt_payload(event_payload or {})
    if prompt_payload:
        report.merge(validate_langfuse_prompt_payload(manifest, prompt_payload))
    return report


def validate_manifest(manifest: Dict[str, Any]) -> PromptGateReport:
    report = PromptGateReport()
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        report.errors.append("prompts manifest must contain a non-empty prompts list")
        return report

    seen_names = set()
    for index, prompt_spec in enumerate(prompts):
        if not isinstance(prompt_spec, dict):
            report.errors.append(f"prompts[{index}] must be an object")
            continue
        name = str(prompt_spec.get("name", "")).strip()
        if not name:
            report.errors.append(f"prompts[{index}].name is required")
            continue
        if name in seen_names:
            report.errors.append(f"duplicate prompt manifest entry: {name}")
        seen_names.add(name)
        report.checked_prompts.append(name)
        report.merge(_validate_prompt_spec(prompt_spec))
    return report


def validate_langfuse_prompt_payload(
    manifest: Dict[str, Any],
    prompt_payload: Dict[str, Any],
) -> PromptGateReport:
    report = PromptGateReport()
    name = str(prompt_payload.get("name", "")).strip()
    if not name:
        report.errors.append("Langfuse prompt payload is missing prompt.name")
        return report

    prompt_spec = _find_prompt_spec(manifest, name)
    if prompt_spec is None:
        report.warnings.append(f"prompt {name} is not managed by prompts/manifest.json; skipped")
        return report

    report.checked_prompts.append(name)
    required_variables = set(_string_list(prompt_spec.get("required_variables")))
    prompt_variables = extract_prompt_variables(prompt_payload.get("prompt"))
    missing_variables = sorted(required_variables - prompt_variables)
    if missing_variables:
        report.errors.append(
            f"prompt {name} is missing required variables: {', '.join(missing_variables)}"
        )

    report.merge(_validate_prompt_config(name, prompt_spec, prompt_payload.get("config")))
    labels = set(_string_list(prompt_payload.get("labels")))
    release_labels = set(_string_list(prompt_spec.get("release_labels")))
    if labels.intersection(release_labels):
        gates = prompt_spec.get("quality_gates")
        if not isinstance(gates, dict) or not gates.get("dataset") or not gates.get("metrics"):
            report.errors.append(
                f"prompt {name} has release label but no quality_gates.dataset/metrics configured"
            )
    return report


def extract_langfuse_prompt_payload(event_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not event_payload:
        return None
    if isinstance(event_payload.get("prompt"), dict):
        return event_payload["prompt"]
    client_payload = event_payload.get("client_payload")
    if isinstance(client_payload, dict) and isinstance(client_payload.get("prompt"), dict):
        return client_payload["prompt"]
    return None


def extract_prompt_variables(prompt: Any) -> set[str]:
    return {
        variable
        for text in _prompt_text_fragments(prompt)
        for variable in PROMPT_VARIABLE_PATTERN.findall(text)
    }


def _validate_prompt_spec(prompt_spec: Dict[str, Any]) -> PromptGateReport:
    report = PromptGateReport()
    name = str(prompt_spec.get("name", "")).strip()
    prompt_type = prompt_spec.get("type")
    if prompt_type not in {"chat", "text"}:
        report.errors.append(f"prompt {name} type must be chat or text")

    required_variables = set(_string_list(prompt_spec.get("required_variables")))
    if not required_variables:
        report.errors.append(f"prompt {name} must declare required_variables")

    fallback_variables = extract_prompt_variables(prompt_spec.get("fallback"))
    missing_fallback_variables = sorted(required_variables - fallback_variables)
    if missing_fallback_variables:
        report.errors.append(
            f"prompt {name} fallback is missing variables: {', '.join(missing_fallback_variables)}"
        )

    gates = prompt_spec.get("quality_gates", {})
    if gates:
        report.merge(_validate_quality_gates(name, gates))
    return report


def _validate_quality_gates(name: str, gates: Any) -> PromptGateReport:
    report = PromptGateReport()
    if not isinstance(gates, dict):
        report.errors.append(f"prompt {name} quality_gates must be an object")
        return report
    if not str(gates.get("dataset", "")).strip():
        report.errors.append(f"prompt {name} quality_gates.dataset is required")
    metrics = gates.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        report.errors.append(f"prompt {name} quality_gates.metrics must be a non-empty object")
        return report
    for metric_name, threshold in metrics.items():
        if not isinstance(threshold, dict) or "min" not in threshold:
            report.errors.append(f"prompt {name} metric {metric_name} must declare min threshold")
            continue
        try:
            float(threshold["min"])
        except (TypeError, ValueError):
            report.errors.append(f"prompt {name} metric {metric_name} min must be numeric")
    return report


def _validate_prompt_config(
    name: str,
    prompt_spec: Dict[str, Any],
    config: Any,
) -> PromptGateReport:
    report = PromptGateReport()
    if config is None:
        return report
    if not isinstance(config, dict):
        report.errors.append(f"prompt {name} config must be an object")
        return report

    schema = prompt_spec.get("config_schema") if isinstance(prompt_spec, dict) else {}
    allowed_keys = set(_string_list((schema or {}).get("allowed_keys")))
    unknown_keys = sorted(set(config.keys()) - allowed_keys) if allowed_keys else []
    if unknown_keys:
        report.errors.append(f"prompt {name} config has unsupported keys: {', '.join(unknown_keys)}")

    if "temperature" in config and not _is_number_between(config["temperature"], 0, 2):
        report.errors.append(f"prompt {name} config.temperature must be between 0 and 2")
    if "max_tokens" in config and not _is_positive_integer(config["max_tokens"]):
        report.errors.append(f"prompt {name} config.max_tokens must be a positive integer")
    if "model" in config and not str(config["model"]).strip():
        report.errors.append(f"prompt {name} config.model must be a non-empty string")
    return report


def _find_prompt_spec(manifest: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for prompt_spec in manifest.get("prompts", []):
        if isinstance(prompt_spec, dict) and prompt_spec.get("name") == name:
            return prompt_spec
    return None


def _prompt_text_fragments(prompt: Any) -> Iterable[str]:
    if isinstance(prompt, str):
        yield prompt
        return
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    yield content


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_number_between(value: Any, lower: float, upper: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return lower <= numeric <= upper


def _is_positive_integer(value: Any) -> bool:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return False
    return numeric > 0
