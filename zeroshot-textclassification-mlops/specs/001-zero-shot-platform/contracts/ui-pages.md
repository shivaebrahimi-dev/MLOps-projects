# Contract: Streamlit Pages

Pages contain interface logic only. They call services and show the results or errors those services return.

## Admin page (`app/pages/1_Admin.py`)

Shows a notice that the page is intended only for local development or trusted environments.

| Area | Elements | Behavior |
|---|---|---|
| Models | Table of name, identifier, status, revision, and path | Shows every configured model. |
| Locations | Model directory and cache location | Read-only. |
| Download | Multi-select and Download button | Reports each model's outcome. Ready models are reported as already available. |
| Delete | Model selector, confirmation checkbox, and Delete button | Button stays disabled until confirmed. |
| Cache | Confirmation checkbox and Clear Cache button | Reports what was removed, or that no cache exists. |
| Refresh | Refresh button | Reloads model status from disk. |
| Labels | Set selector, add and remove controls, and Save button | Validation errors appear next to the form. Saved sets survive restarts. |

## Classification page (`app/pages/2_Classify_Text.py`)

| Area | Elements | Behavior |
|---|---|---|
| Input | Text area | Shows the character limit. Text is displayed as plain text only. |
| Selection | Ready-model selector and label-set selector | When none are available, shows what is missing and points to the admin page. |
| Submit | Classify button | Shows progress while running. |
| Result | Input text, predicted class, confidence, ranked table, model, revision, timestamp, and request ID | Scores are rounded to two decimals for display. |
| Storage | Success or error message | On success, shows the bucket and object path. On failure, shows the result with a "not saved" error and the request ID. |

## Error messages

Every error states what happened and what to do next, and never includes secrets or stack traces.

| Situation | Required message content |
|---|---|
| Empty text | Asks the user to enter text. |
| Text too long | States the limit and the submitted length. |
| No ready model | Directs the user to download a model on the admin page. |
| Invalid label set | States that at least two unique labels are needed. |
| Download failed | Names the model and the likely cause, and states that it can be retried. |
| Storage unavailable | States that MinIO could not be reached, that the result was not saved, and how to start MinIO. |
