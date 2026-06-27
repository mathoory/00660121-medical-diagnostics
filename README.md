# 00660121 – Medical Diagnostics (AI in Biotech)  
Technion – Spring 2026

Welcome to the course repository for **Medical Diagnostics**, a 4-week module introducing deep and machine learning for biotechnology students. This repo contains all materials needed for the tutorials and assignments.

---

## 📚 Course Structure

| Week | Topic                                 | Focus                         |
|------|---------------------------------------|-------------------------------|
| 1    | Regression & Gradient Descent         | Fitting models to data        |
| 2    | Neural Networks (MLP)                 | Perceptron and hidden layers  |
| 3    | CNNs & Image Classification           | Features in biological images |
| 4    | Transformers & Nanopore Sequencing    | Sequences & base-calling      |


---

## 📌 Guidelines

* **Academic integrity**: Cheating is not tolerated. Code must not be copied, shared, or transferred in any way. Use of LLMs (e.g. ChatGPT) is encouraged, but you are fully responsible for verifying correctness. Prompts and generated code should not be shared between students.

* **Assignments**: There are two planned homework assignments and one final project.

* **Grading**: Based on the written report.

* **Homework support**: All questions must be posted in the relevant Moodle forum. Forum answers are considered official clarifications—review them before submission.

* **Appeals & personal issues**: May be sent directly to the TA by email.

---

## 🛠️ Repository Usage

Students should use the main branch: click Code → Download ZIP, or `git clone` to get the latest version. 
Fixes are pushed to main over the semester, so re-download or pull if something was updated. Solutions are not included. The dev branch is for development and is safe to ignore.

``` 
git switch main
git checkout dev -- path/to/hw1        # only these paths cross over (already staged)
git commit -m "Release HW1 (v2): fix Q3"
git switch dev                          # back to work
```