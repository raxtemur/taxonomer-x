# TaxoBranch paper

`main.tex` is the anonymous ACL Rolling Review source and `main.pdf` is the
compiled manuscript. Generated numeric macros under `generated/` are exported
from validated experiment manifests; values are not transcribed manually.

Build with:

```bash
tectonic main.tex --keep-logs
```

Release checks:

- anonymous review mode is enabled;
- the abstract is below 200 words;
- the numbered main paper, including limitations and ethics, ends on page 8;
- references begin on page 8 and supplementary material begins on page 9;
- the architecture diagram is vector TikZ source;
- matched-objective, non-inferiority, and stress-test values come from separate
  non-pooled retraining sets.

Keep author names and acknowledgments out of this file until camera-ready.
