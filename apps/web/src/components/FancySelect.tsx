import { ChevronDown } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

type FancySelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
  hint?: string;
};

type FancySelectProps = {
  value: string;
  options: FancySelectOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  menuClassName?: string;
  buttonClassName?: string;
};

export function FancySelect({
  value,
  options,
  onChange,
  placeholder = "请选择",
  className,
  menuClassName,
  buttonClassName,
}: FancySelectProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const selected = useMemo(() => options.find((item) => item.value === value), [options, value]);

  useEffect(() => {
    function onPointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, []);

  return (
    <div className={`fancy-select ${className ?? ""}`} ref={rootRef}>
      <button
        aria-expanded={open}
        className={`fancy-select-trigger ${buttonClassName ?? ""}`}
        onClick={() => setOpen((prev) => !prev)}
        type="button"
      >
        <span className={`fancy-select-value ${selected ? "" : "placeholder"}`}>
          {selected?.label ?? placeholder}
        </span>
        <ChevronDown className={open ? "open" : ""} size={14} />
      </button>

      {open && (
        <div className={`fancy-select-menu ${menuClassName ?? ""}`} role="listbox">
          {options.map((option) => (
            <button
              className={`fancy-select-option ${option.value === value ? "active" : ""}`}
              disabled={option.disabled}
              key={option.value}
              onClick={() => {
                if (option.disabled) return;
                onChange(option.value);
                setOpen(false);
              }}
              type="button"
            >
              <span>{option.label}</span>
              {option.hint && <small>{option.hint}</small>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
