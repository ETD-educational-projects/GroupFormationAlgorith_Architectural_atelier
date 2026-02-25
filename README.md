# Student Group Formation System

An intelligent algorithm for forming balanced student study groups based on survey responses, optimizing both similarity and diversity metrics.

## Features

- **Similarity-based matching**: Groups students with similar preferences and characteristics
- **Diversity optimization**: Ensures groups have diverse skill sets and perspectives
- **Strategic leader selection**: Identifies potential group leaders using a multi-step selection process
- **Greedy assignment algorithm**: Iteratively assigns students to groups to maximize group cohesion
- **Composition analysis**: Provides visualizations of group balance and diversity metrics

## Algorithm Overview

### 1. Similarity Matrix
Compares student responses across similarity questions. Students with identical answers receive a score of 1.0 (fully similar) for that question.

### 2. Diversity Matrix
Evaluates differences in student characteristics across diversity questions:
- **Set comparisons**: Calculates symmetric difference / union for multi-select questions
- **Numeric values**: Uses normalized distance for rating scales
- **Categorical**: Direct comparison for single-select responses

### 3. Leader Selection
The algorithm selects initial group leaders through a three-step process:
1. **First leader**: Randomly selected from candidates with "group leader" role
2. **Second leader**: Selected for maximum diversity from the first leader
3. **Third leader**: Selected for minimum similarity to the first leader

### 4. Group Assignment
Uses greedy assignment to place remaining students:
- For each unassigned student, finds the best-fit group (highest average affinity)
- Tie-breaking prefers smaller groups to balance group sizes
- Respects maximum group size constraint

## Input Format

The input CSV file must:
- Include an `ID` column with unique student identifiers
- Include survey response columns (Q1, Q2, Q3, etc.)
- Include a role question column (default: Q14) to identify potential leaders

### Column Mapping
Columns are automatically renamed from their original names to Q1, Q2, Q3, etc., in sequential order.

## Output

The notebook generates:
- **CSV file**: Student list with assigned group column
- **Group lists**: Console output showing final group compositions
- **Analysis tables**: Distribution of characteristics across groups

## Configuration

Edit the following variables in the first cell to customize the algorithm:

```python
file_name = "V1.csv"              # Input CSV file name
index_column = 'ID'               # Column name for student IDs
role_question = "Q14"             # Question identifying group leaders
leader_answer = "Taking charge and organizing tasks (group leader)"  # Leader role text
software_question = "Q11"         # Question about software/tools
SIMILARITY_QUESTIONS = {5, 6, 7, 8, 9, 10, 15, 16} 
DIVERSITY_QUESTIONS = {3, 4, 11, 12, 13, 14}
max_group_size = 15               # Maximum students per group
group_column = "Group_columns"    # Name of output group column
output_csv = "test.csv"           # Output CSV file name
```

## Requirements

- Python 3.7+
- pandas
- numpy

## Usage

1. Prepare your survey data in CSV format (semicolon-separated)
2. Update the `file_name` variable to point to your CSV
3. Run all notebook cells in order
4. Review group compositions in console output
5. Analyze group balance using the provided analysis tables
6. Export results using the generated CSV file

## Example

See `Group_formation.ipynb` for a complete working example.

## Notes

- The algorithm is non-deterministic due to random leader selection; run multiple times for different groupings
- Set `max_group_size` to `None` for unlimited group sizes
- The similarity and diversity matrices are normalized (0-1 scale)
- Affinity scores combine both similarity and diversity (sum of both matrices)

## License

This project is licensed under the MIT License - see below for details.

### MIT License

```
MIT License

Copyright (c) 2026 Giacomo Chiesa, Ali JahaniRahaei

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Authors

Giacomo Chiesa - Politecnico di Torino - Giacomo.chiesa@polito.it
Ali JahaniRahaei - Politecnico di Torino - Ali.jahanirahaei@polito.it