# CLAUDE.md — Paper Writing Assistant

## Project

We are writing a 7-page workshop paper for COLM titled:

**"Don't Paste In Conditioning Tokens: Noising Conditioning Tokens for Conditional Generation in Continuous Diffusion Language Models"**

## Objective

Collaboratively draft and revise `main.tex` toward a complete, submission-ready paper.

---

## Core Rules

### Propose diffs, never apply unilaterally

Always show proposed changes as clearly marked blocks before editing any file:

- Use `[REPLACE]` to show text being removed
- Use `[WITH]` to show the replacement
- Use `[ADD AFTER: "..."]` to show insertions

Wait for explicit approval ("apply", "looks good", "go ahead") before modifying `main.tex` or any file.

### Ground all writing in evidence

Before drafting any claim, result, or comparison, read the relevant file in `paper/notes/`.
Every quantitative claim must trace to a specific entry in the notes.
If a number is not in the notes, say so — do not estimate or interpolate.

### Never hallucinate references

Do not invent paper titles, author names, venue names, or years.
If a citation is needed and you don't have it, write `\cite{TODO}` with a comment explaining
what kind of work should go there.

When the human supplies an arXiv link, fetch the page, extract the correct title, authors,
and year, and add a proper BibTeX entry to `references.bib`.

### LaTeX hygiene

After any approved edit is applied, run `latexmk -pdf main.tex` to confirm clean compilation.
If compilation fails, read the error output and propose a fix before stopping.

---

## Paper Structure (target: 7 pages, COLM workshop format)

1. Abstract
2. Introduction
3. Background / Related Work
4. Method
5. Experiments
   - Sudoku (Easy)
   - N-Queens (8x8, 10x10)
6. Analysis / Discussion
7. Conclusion

---

## File Layout

```
paper/
  main.tex
  references.bib
  notes/          # source of truth for results and experimental logs
  figures/        # plots, if any
```

---

## Diff Format

Show proposed changes as readable blocks, not unified diffs. Example:

```
[REPLACE]
We evaluate on two tasks.

[WITH]
We evaluate on three structured reasoning tasks.
```

```
[ADD AFTER: "\end{abstract}"]
\section{Introduction}
...
```

---

## Workflow Per Session

1. Human provides a prompt describing what to work on
2. Claude reads current `main.tex` and relevant files in `paper/notes/`
3. Claude proposes changes using the diff format above
4. Human approves, requests changes, or edits directly
5. Claude applies approved changes
6. Claude runs `latexmk -pdf main.tex` and confirms clean compilation
7. Human commits checkpoint to git