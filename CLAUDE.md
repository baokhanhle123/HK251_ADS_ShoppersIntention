# CLAUDE.md - Project Instructions for Claude Code

## Interaction Guidelines

**CRITICAL: Discussion Before Execution**

- ALWAYS ask and discuss to find out what I really want
- ASK me if you need clarification - never make unsure assumptions
- WAIT for explicit approval before executing any plan
- Present plans for review before implementation

**Trigger words to execute:**
- "go"
- "execute your plan"
- "do it"
- "let's go"

Until you hear one of these, keep planning and discussing.

---

## Planning Workflow

1. **Understand** - Ask clarifying questions to understand my intent
2. **Plan** - Create a detailed execution plan
3. **Review** - Present the plan for my approval
4. **Wait** - Do NOT execute until I say "go" or "execute your plan"
5. **Execute** - Only after explicit approval, proceed step by step

---

## Project Context

| Property | Value |
|----------|-------|
| Course | HK251 - Applied Data Science |
| Project | Online Shoppers Purchasing Intention Analysis |
| Business Narrative | ShopSmart E-commerce (fictional Vietnamese platform) |
| Primary Tool | RapidMiner Studio (NOT Python) |
| Report Language | Vietnamese with English technical terms |
| Dataset | UCI Online Shoppers Purchasing Intention (12,330 sessions) |

---

## Project Structure

```
HK251_ADS_ShoppersIntention/
├── Report/                     # LaTeX report (Vietnamese)
│   ├── main.tex               # Main document with title page
│   ├── sections/
│   │   ├── 1_Introduction.tex # Business context, problem statement
│   │   ├── 2_Methods.tex      # Data understanding & preparation
│   │   ├── 3_Experiments.tex  # Modeling & evaluation
│   │   ├── 4_Improvement.tex  # Discussion & recommendations
│   │   └── 5_Conclusion.tex   # Summary & future work
│   └── references.bib         # Bibliography
├── Project_Requirement/        # Course requirements (Vietnamese)
├── ADS_Assignment.ipynb        # Reference only (Python implementation)
├── sakar2018.pdf              # Original research paper
├── README.md                  # Project documentation
└── CLAUDE.md                  # This file
```

---

## Editing Guidelines

### LaTeX Files
- Write in Vietnamese with English technical terms in parentheses
- Example: "Cây quyết định (Decision Tree)"
- Use figure placeholders: `[Hình X: Description]`
- Use table placeholders: `[Bảng X: Description]`
- Follow CRISP-DM structure throughout

### RapidMiner References
- Always specify operator names: `Decision Tree`, `Random Forest`, `k-NN`, `Logistic Regression`
- Include parameter settings when relevant
- Reference SMOTE and Random Oversampling for class imbalance

### Communication Style
- Be concise but thorough
- Track tasks using TodoWrite tool
- Report progress clearly after each step
- Summarize what was done at the end

---

## Avoid

- Making changes without discussion first
- Assuming requirements when unclear
- Executing before explicit approval
- Creating unnecessary files or documentation
- Over-engineering solutions
- Adding features not requested
- Guessing what I want instead of asking

---

## Quick Reference

```
User says: "Let's do X"
Claude: Asks clarifying questions, creates plan, presents for review

User says: "go" / "execute"
Claude: Proceeds with execution step by step

User says: "stop" / "wait"
Claude: Pauses and asks what to do next
```
