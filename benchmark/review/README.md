# Review 150 instrument questions

Open [quick-review.html](quick-review.html) for rapid 1–5 quality ratings and
comments, or [review.html](review.html) for the detailed independent validity
review. Both are standalone browser files. Download the HTML to your computer
and open it directly; no cluster server or model is needed.

In the quick form, use 1–5 to score and the arrow keys to move between questions
when the cursor is outside a text field. Scores do not advance automatically.
Add an opinion and optionally select acceptable answers. Export progress JSON
regularly; browser storage alone is not a reliable backup. Import a previous
JSON export to continue, especially after moving the HTML file.

For the detailed review, independently judge relevance, clarity, acceptable
answers, answerability and fault realism under the stated simulator assumptions.
Record your reviewer ID and date. Download the ratings CSV and progress JSON.
Quick quality scores are supplemental and do not replace this validity review.

The 150 public cases exactly match the current frozen 388-question release.
Slurm check `4636471` verified task IDs, public question content, both embedded
HTML payloads and source fingerprints. [compatibility.json](compatibility.json)
records that evidence. No reference answers are embedded in these HTML files.
The blank `ratings.csv` is a template; no independent human assessment is
represented as completed. Existing progress on your browser is separate from
that template and should be preserved with an export.

The source question hash is
`b720504d739c5f32a5677d4964bf60591ed48a645d3627bc85ede61af5398c04`.
Keep the review files out of agent inputs and retain the exported human ratings
separately from model predictions. Simulation consistency and human validity
are distinct checks.
