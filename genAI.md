# GenAI Usage Report

## Tool Used
- **Claude (Anthropic)** - claude.ai web interface with computer use capabilities (Model - Opus 4.5)

---

## Prompting Strategy

We followed an **iterative refinement and verification strategy**,

1. **Task Specification**: Provided the model with clear goals
2. **Domain Context**: Referenced the specific paper (arXiv:1910.06242) and shared the specific code/notebook content that is under construction
3. **Verification**: Asked Claude to explain complex concepts back to us to verify correctness and improve understanding
4. **Error Correction**: When we spotted mistakes, we challenged Claude's output and asked for corrections and refinements
5. **Learning Through Examples**: Request concrete numerical examples to build intuition

This strategy aligns with the "Chain of Verification" and "Socratic Prompting" approaches, where we actively questioned outputs rather than accepting them blindly.

---

## Specific Uses and Outcomes

### 1. Slide Creation for Presentation

**Prompts**: 
- "Create Slides on paper methodology - entropy, eigenvalues, correlation matrix"
- "Insert the formula for calculating C_M and C_GR into the slides"
- "Make a slide specifying the 9 crash events used for analysis"

**Output**: PowerPoint slides (4 slides total) explaining the paper's methodology

**What we adopted**: The slide structure and visual layout

**What we corrected**: 
- Claude initially defined H_M as "log(N)" (maximum entropy) and H_GR as "log(N) - log(2)" (golden ratio). We caught this error by cross-referencing with the paper and the notebook code. The correct definitions were:
  - H_M = entropy computed from market mode matrix C_M
  - H_GR = entropy computed from group-random mode matrix C_GR
- Claude corrected the slides after we pointed out the error

**Lesson learned**: GenAI can confidently present incorrect information. Domain knowledge is essential for verification.

---

### 2. Understanding the concepts used in the paper

**Prompt Examples**:
- "Explain how centrality of stock v_i is calculated after computing v, the eigenvector"
- "How is C_M calculated using spectral decomposition?"
- "How is the eigen centrality distribution plotted for different ranks?"

**Output**: Detailed explanations with mathematical notations and worked examples for better understanding.

**What we adopted**: The conceptual explanations which we later used for verifying if the code is correctly implementing the logic.

**What we verified**: Cross-checked the explained concepts and defintions from the paper and other external sources.

**Example Key insight gained**:
1. Eigendecomposition of C (correlation matrix) → to get C_M and C_GR
2. Eigendecomposition of A = |C|² → to get eigenvector centrality

Claude's examples helped simplify the task of reading the entire paper and helped in understanding every concept in a short span of time.

---

### 3. Generating new code and Debugging existing code Issues

**Input**: Gave the model specific tasks that we needed to implement or the erroneous/faulty code that needed debugging

**Output**: Implementation/Diagnosis of the code along with the errors and their fixes

**What we adopted**: 
- Verified if the output code was implemented correctly by comparing it with theory and checking if the code fixed the errors, before using it in our notebook

**What we modified**: Used Claude's suggested approach but simplified it for our specific use case (models can sometimes complicate
simple code implementations)

---

### 4. Plot Replication and Marker Annotation

**Input**:
We used the model to generate plotting code for **specific figures** in our analysis. Tasks were limited to producing plot layouts and adding visual elements such as centroids, crash-day markers, trajectory lines, vertical reference lines, and IQR shading.

**Output**:
Matplotlib-based code for PCA phase-space plots and time-aligned Mahalanobis distance plots with appropriate markers, annotations, and legends.

**What we adopted**:

* Plot structure and marker placement logic
* Visual conventions used to highlight pre-crash states and crash onsets

**What we verified and modified**:

* Verified that all plotted quantities, alignments, and aggregations matched the underlying data and definitions
* Simplified and adjusted the generated code to fit our notebook and ensure consistency across figures

**Scope limitation**:
GenAI was used **only** for initial plot production and visual annotation; interpretation and validation were done independently.

---

## Effectiveness and Limitations

### Strengths
1. **Rapid prototyping**: Slide creation that would take hours was done in minutes
2. **Code generation**: Debugging suggestions and code snippets were helpful starting points
3. **Explanation generation**: Breaking down complex concepts with examples aided understanding
4. **Iterative refinement**: Claude responded well to corrections and produced improved outputs

### Limitations
1. **Confident incorrectness**: Claude presented wrong definitions with full confidence. Without domain knowledge, we would have used incorrect information.
2. **Context dependency**: Claude's quality depends heavily on how well we specify the problem. Vague prompts led to generic or incorrect outputs.
3. **Verification burden**: Every output required manual verification against the paper.

---

## Conclusion

GenAI is valuable as a **productivity accelerator** and **explanation generator**, but requires constant vigilance. The most important lesson, GenAI outputs must be treated as hypotheses to verify, not facts to accept. Our domain knowledge from reading the paper was essential for catching errors that Claude presented confidently.
The iterative correction process, where we challenged Claude's outputs and asked for explanations, was more effective than accepting first responses. 
