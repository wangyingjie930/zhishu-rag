import {
  CHUNKING_FIELD_HELP,
  CHUNKING_FIELD_LABELS,
  CHUNKING_STRATEGY_HELP,
  CHUNKING_STRATEGY_OPTIONS,
  CHUNKING_STRATEGY_FIELDS,
  DEFAULT_CHUNKING_POLICY,
} from "../lib/chunkingPolicy";
import type { ChunkingPolicyField, ChunkingPolicyInput } from "../lib/chunkingPolicy";
import { FancySelect } from "./FancySelect";
import { useEffect, useState } from "react";

type ChunkingPolicyFieldsProps = {
  value?: ChunkingPolicyInput;
  onChange?: (value: ChunkingPolicyInput) => void;
};

export function ChunkingPolicyFields({ value, onChange }: ChunkingPolicyFieldsProps) {
  const [localPolicy, setLocalPolicy] = useState<ChunkingPolicyInput>(DEFAULT_CHUNKING_POLICY);

  const policy = value ?? localPolicy;

  useEffect(() => {
    if (value) setLocalPolicy(value);
  }, [value]);

  const strategyOptions = CHUNKING_STRATEGY_OPTIONS.map((option) => ({
    value: option.value,
    label: option.label,
    hint: CHUNKING_STRATEGY_HELP[option.value],
  }));
  const activeFields = CHUNKING_STRATEGY_FIELDS[policy.strategy] ?? [];

  function updatePolicy(next: Partial<ChunkingPolicyInput>) {
    const merged = { ...policy, ...next };
    setLocalPolicy(merged);
    onChange?.(merged);
  }

  function updateNumberField(field: keyof ChunkingPolicyInput, rawValue: string) {
    updatePolicy({ [field]: Number(rawValue) } as Partial<ChunkingPolicyInput>);
  }

  return (
    <div className="chunking-policy-fields">
      <FancySelect
        onChange={(nextStrategy) =>
          updatePolicy({ strategy: nextStrategy as typeof DEFAULT_CHUNKING_POLICY.strategy })
        }
        options={strategyOptions}
        value={policy.strategy}
      />
      <input name="strategy" type="hidden" value={policy.strategy} />

      <div className="chunking-grid">
        {activeFields.map((field) => (
          <PolicyNumberField
            field={field}
            key={field}
            onChange={updateNumberField}
            value={policy[field]}
          />
        ))}
      </div>

      <div className="chunking-help-panel">
        <p>{CHUNKING_STRATEGY_HELP[policy.strategy]}</p>
        <div className="chunking-help-list">
          {activeFields.map((field) => (
            <div className="chunking-help-item" key={field}>
              <strong>{CHUNKING_FIELD_LABELS[field]}</strong>
              <span>{CHUNKING_FIELD_HELP[field]}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PolicyNumberField({
  field,
  onChange,
  value,
}: {
  field: ChunkingPolicyField;
  onChange: (field: keyof ChunkingPolicyInput, rawValue: string) => void;
  value: number;
}) {
  const inputConfig = getNumberInputConfig(field);

  return (
    <label className="compact-field" title={CHUNKING_FIELD_HELP[field]}>
      <span>{CHUNKING_FIELD_LABELS[field]}</span>
      <input
        max={inputConfig.max}
        min={inputConfig.min}
        name={field}
        onChange={(event) => onChange(field, event.currentTarget.value)}
        step={inputConfig.step}
        type="number"
        value={value}
      />
    </label>
  );
}

function getNumberInputConfig(field: ChunkingPolicyField) {
  if (field === "overlap_ratio") return { min: 0.1, max: 0.2, step: 0.01 };
  if (field === "semantic_threshold") return { min: 80, max: 98, step: 1 };
  if (field === "semantic_buffer_size") return { min: 1, max: 4, step: 1 };
  if (field === "window_size") return { min: 1, max: 4, step: 1 };
  return { min: 50, max: field === "max_chunk_size" ? 2400 : 1600, step: 50 };
}
