# Examples

Synthetic problems to try pirilib on. All quantities, distances and costs are
invented; none come from a company or from competition data.

| Folder | What it shows |
|---|---|
| [`linehaul_mini`](linehaul_mini) | A small line-haul problem with its tables in CSV files, used in the user guide and the tests |
| [`objectives`](objectives) | How the `objective:` section of a problem changes the plan |

Run them from the repository root:

```bash
pirilib solve examples/linehaul_mini/problem.yaml -o plan.csv
pirilib check examples/linehaul_mini/problem.yaml plan.csv
```
