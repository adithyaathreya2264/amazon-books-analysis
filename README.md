# Amazon Best-Selling Books Data Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A comprehensive data analysis project exploring Amazon's best-selling books from 2009-2019, uncovering trends in genres, authors, ratings, reviews, pricing, and temporal patterns using Python and data science libraries.

---

## Project Overview

This project performs an end-to-end data analysis of Amazon best-selling books to answer critical business questions:

- What genres dominate the best-seller list?
- Which authors have the most best-sellers?
- How do ratings vary across genres?
- What's the relationship between review counts and ratings?
- How are best-sellers priced?
- What are the year-over-year trends?
- Does higher price correlate with better ratings?

The analysis follows industry-standard data science practices including data cleaning, exploratory data analysis (EDA), statistical analysis, and data visualization.

---

## Project Structure

```
amazon-books-analysis/
│
├── data/
│   └── amazon_books.csv          # Dataset with 550 Amazon best-selling books
│
├── src/
│   └── analysis.py                # Complete analysis script with OOP design
│
├── output/                        # Generated visualizations and insights
│   ├── genre_distribution.png
│   ├── rating_by_genre.png
│   ├── top_authors.png
│   ├── reviews_vs_ratings.png
│   ├── price_distribution.png
│   ├── yearly_trends.png
│   ├── price_vs_rating.png
│   └── insights.txt
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation (this file)
└── venv/                          # Virtual environment (excluded from git)
```

---

## Dataset Description

**Source**: Amazon Top 50 Bestselling Books (2009-2019)  
**Records**: 550 books  
**File**: `data/amazon_books.csv`

### Columns

| Column        | Type    | Description                               |
|---------------|---------|-------------------------------------------|
| Name          | String  | Book title                                |
| Author        | String  | Book author                               |
| User Rating   | Float   | Average user rating (0-5 scale)           |
| Reviews       | Integer | Total number of user reviews              |
| Price         | Integer | Book price in USD                         |
| Year          | Integer | Year of best-seller status                |
| Genre         | String  | Book category (Fiction / Non Fiction)     |

---

## Tools & Libraries

### Core Technologies
- **Python 3.10+** - Programming language
- **pandas 2.0+** - Data manipulation and analysis
- **NumPy 1.24+** - Numerical computations
- **Matplotlib 3.7+** - Data visualization
- **Seaborn 0.12+** - Statistical data visualization

### Development Environment
- **Virtual Environment** - Isolated dependency management
- **Object-Oriented Programming** - Clean, modular code design

---

## How to Run the Project

### Prerequisites
- Python 3.10 or higher installed
- Basic familiarity with command line/terminal

### Step 1: Clone or Download the Project

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Analysis

```bash
python src/analysis.py
```

### Step 6: View Results

After execution, check the `output/` directory for:
- **7 visualization images** (PNG format, 300 DPI)
- **insights.txt** - Text file with key findings

---

## Analysis Workflow

The analysis follows a **6-phase pipeline**:

### Phase 1: Environment Setup & Data Loading
- Import required libraries
- Load dataset from CSV
- Verify data integrity

### Phase 2: Data Understanding
- Display first 5 rows (`.head()`)
- Show dataset info (`.info()`)
- Generate statistical summary (`.describe()`)
- Understand data types and structure

### Phase 3: Data Cleaning
- Rename columns to `snake_case` convention
- Convert data types (ensure numeric columns are correct)
- Check and handle missing values
- Detect and remove duplicates
- Perform sanity checks (rating ranges, negative values)

### Phase 4: Exploratory Data Analysis (EDA)

**7 Key Questions Answered:**

1. **Genre Distribution** - How many books in each genre?
2. **Rating by Genre** - Which genre has better ratings?
3. **Top Authors** - Who are the top 10 authors by best-seller count?
4. **Reviews vs Ratings** - Do more reviews mean higher ratings?
5. **Price Distribution** - How are best-sellers priced?
6. **Yearly Trends** - How have best-sellers evolved over time?
7. **Price vs Rating** - Does expensive mean better quality?

### Phase 5: Visualizations

All questions are accompanied by professional visualizations:
- **Bar charts** - Genre distribution, top authors
- **Box plots** - Rating distributions
- **Scatter plots** - Correlation analysis
- **Histograms** - Price distribution
- **Line charts** - Temporal trends

**Visualization Standards:**
- High resolution (300 DPI)
- Proper titles and axis labels
- Clean styling with consistent color schemes
- Saved as PNG files for easy sharing

### Phase 6: Insights & Conclusions

The analysis generates **7 key business insights** including:
- Genre preferences and market balance
- Quality standards of best-sellers
- Author dominance patterns
- Pricing strategies
- Review-rating dynamics
- Temporal market trends
- Price-quality relationships

---
## Key Insights

> **Note**: Run the analysis to generate current insights. Below are example findings:
1. **Genre Balance**: The dataset shows a distribution between Fiction and Non-Fiction books, indicating diverse reading preferences on Amazon.
2. **High Quality Standard**: The average rating is 4.6+/5.0, demonstrating that best-selling books maintain exceptional quality standards.
3. **Author Dominance**: Certain authors appear multiple times, showing strong brand recognition and reader loyalty.
4. **Affordable Pricing**: The median price is under $15, suggesting best-sellers are priced accessibly to reach wider audiences.
5. **Reviews vs Ratings**: The correlation analysis reveals interesting patterns about book popularity and quality perception.
6. **Temporal Patterns**: Year-wise trends show fluctuations in best-seller counts, reflecting market dynamics.
7. **Price vs Quality**: Price correlation with rating is weak, indicating readers value content over cost.
---

## Future Improvements

- **Sentiment Analysis**: Analyze actual review text for deeper insights
- **Author Network**: Build a network graph of co-authorship and genre crossovers
- **Predictive Modeling**: Train ML models to predict best-seller potential
- **Time Series Forecasting**: Predict future trends using ARIMA/Prophet
- **Interactive Dashboard**: Create a Streamlit/Dash web app for dynamic exploration
- **Expand Dataset**: Include more years and additional features (publisher, page count)
- **Genre Deep Dive**: Analyze sub-genres and cross-genre books
- **Regional Analysis**: Compare best-sellers across different Amazon markets

---

## Code Quality Features

- **Object-Oriented Design** - Clean class-based architecture
- **Comprehensive Documentation** - Docstrings for all methods
- **Type Hints** - Clear parameter types
- **Error Handling** - Robust exception management
- **Modular Functions** - Single responsibility principle
- **Professional Output** - Formatted console logs with visual separators
- **Reproducibility** - Consistent results across runs

---

## Learning Outcomes

This project demonstrates proficiency in:

- **Data Wrangling**: Loading, cleaning, and transforming real-world data
- **Statistical Analysis**: Computing correlations, distributions, and aggregations
- **Data Visualization**: Creating publication-quality charts
- **Python Programming**: OOP, pandas operations, file I/O
- **Analytical Thinking**: Translating data into business insights
- **Project Organization**: Maintaining clean directory structure
- **Documentation**: Writing clear README and code comments


Built by: Adithya R Athreya