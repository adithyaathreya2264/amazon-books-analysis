"""
Amazon Best-Selling Books Data Analysis
==========================================
A comprehensive analysis of Amazon best-selling books (2009-2019)
exploring trends in genres, authors, ratings, reviews, pricing, and yearly patterns.

Author: Data Analysis Project
Date: 2025-12-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class AmazonBooksAnalyzer:
    """Main class for Amazon books data analysis"""
    
    def __init__(self, data_path='data/amazon_books.csv'):
        """
        Initialize the analyzer with dataset path
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing Amazon books data
        """
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        
        # Create output directory for visualizations
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """
        Load dataset from CSV file
        
        Returns:
        --------
        pd.DataFrame : Loaded dataframe
        """
        print("="*80)
        print("PHASE 1: LOADING DATA")
        print("="*80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully from {self.data_path}")
            print(f"✓ Dataset contains {len(self.df)} records")
            return self.df
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.data_path}")
            raise
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def understand_data(self):
        """
        Phase 2: Understand the dataset structure and content
        """
        print("\n" + "="*80)
        print("PHASE 2: DATA UNDERSTANDING")
        print("="*80)
        
        print("\n--- First 5 Rows ---")
        print(self.df.head())
        
        print("\n--- Dataset Information ---")
        print(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nColumns: {list(self.df.columns)}")
        
        print("\n--- Data Types and Non-Null Counts ---")
        print(self.df.info())
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        print("\n--- Observations ---")
        print(f"• Dataset contains {self.df.shape[0]} books from Amazon's best-seller list")
        print(f"• Columns include: {', '.join(self.df.columns)}")
        print(f"• Numeric features: User Rating, Reviews, Price, Year")
        print(f"• Categorical features: Name, Author, Genre")
        
    def clean_data(self):
        """
        Phase 3: Clean and prepare data for analysis
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataframe
        """
        print("\n" + "="*80)
        print("PHASE 3: DATA CLEANING")
        print("="*80)
        
        # Create a copy for cleaning
        self.df_clean = self.df.copy()
        
        # Step 1: Rename columns to snake_case
        print("\n--- Step 1: Renaming Columns to snake_case ---")
        column_mapping = {
            'Name': 'name',
            'Author': 'author',
            'User Rating': 'user_rating',
            'Reviews': 'reviews',
            'Price': 'price',
            'Year': 'year',
            'Genre': 'genre'
        }
        self.df_clean.rename(columns=column_mapping, inplace=True)
        print(f"✓ Columns renamed: {list(self.df_clean.columns)}")
        
        # Step 2: Check and convert data types
        print("\n--- Step 2: Data Type Conversions ---")
        print(f"Before conversion:\n{self.df_clean.dtypes}")
        
        # Ensure numeric columns are correct type
        self.df_clean['user_rating'] = pd.to_numeric(self.df_clean['user_rating'], errors='coerce')
        self.df_clean['reviews'] = pd.to_numeric(self.df_clean['reviews'], errors='coerce')
        self.df_clean['price'] = pd.to_numeric(self.df_clean['price'], errors='coerce')
        self.df_clean['year'] = pd.to_numeric(self.df_clean['year'], errors='coerce').astype('Int64')
        
        print(f"\nAfter conversion:\n{self.df_clean.dtypes}")
        print("✓ Data types verified and converted")
        
        # Step 3: Check for missing values
        print("\n--- Step 3: Missing Values Check ---")
        missing = self.df_clean.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "✓ No missing values found")
        
        # Handle missing values if any
        if missing.sum() > 0:
            print("⚠ Handling missing values...")
            # For numeric columns, fill with median
            numeric_cols = ['user_rating', 'reviews', 'price', 'year']
            for col in numeric_cols:
                if self.df_clean[col].isnull().sum() > 0:
                    median_val = self.df_clean[col].median()
                    self.df_clean[col].fillna(median_val, inplace=True)
                    print(f"  • {col}: filled {missing[col]} values with median ({median_val})")
        
        # Step 4: Check for duplicates
        print("\n--- Step 4: Duplicate Detection ---")
        duplicates = self.df_clean.duplicated().sum()
        print(f"Duplicates found: {duplicates}")
        
        if duplicates > 0:
            print("⚠ Removing duplicate rows...")
            self.df_clean.drop_duplicates(inplace=True)
            print(f"✓ Removed {duplicates} duplicate rows")
        else:
            print("✓ No duplicates found")
        
        # Step 5: Sanity checks
        print("\n--- Step 5: Sanity Checks ---")
        
        # Check rating range (should be 0-5)
        invalid_ratings = self.df_clean[(self.df_clean['user_rating'] < 0) | 
                                         (self.df_clean['user_rating'] > 5)]
        print(f"Invalid ratings (not in 0-5 range): {len(invalid_ratings)}")
        
        # Check for negative values
        negative_reviews = self.df_clean[self.df_clean['reviews'] < 0]
        negative_prices = self.df_clean[self.df_clean['price'] < 0]
        
        print(f"Negative review counts: {len(negative_reviews)}")
        print(f"Negative prices: {len(negative_prices)}")
        
        if len(invalid_ratings) == 0 and len(negative_reviews) == 0 and len(negative_prices) == 0:
            print("✓ All sanity checks passed")
        
        print(f"\n--- Cleaning Summary ---")
        print(f"Original records: {len(self.df)}")
        print(f"Clean records: {len(self.df_clean)}")
        print(f"Records removed: {len(self.df) - len(self.df_clean)}")
        
        return self.df_clean
    
    def exploratory_analysis(self):
        """
        Phase 4: Perform Exploratory Data Analysis (EDA)
        """
        print("\n" + "="*80)
        print("PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        # Question 1: Distribution of books by genre
        self._analyze_genre_distribution()
        
        # Question 2: Average user rating per genre
        self._analyze_rating_by_genre()
        
        # Question 3: Top 10 authors
        self._analyze_top_authors()
        
        # Question 4: Relationship between reviews and ratings
        self._analyze_reviews_ratings_relationship()
        
        # Question 5: Price distribution
        self._analyze_price_distribution()
        
        # Question 6: Year-wise trends
        self._analyze_yearly_trends()
        
        # Question 7: Price vs Rating correlation
        self._analyze_price_rating_correlation()
        
    def _analyze_genre_distribution(self):
        """Q1: Distribution of books by genre"""
        print("\n--- Q1: Distribution of Books by Genre ---")
        
        genre_counts = self.df_clean['genre'].value_counts()
        print(genre_counts)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        genre_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title('Distribution of Books by Genre', fontsize=16, fontweight='bold')
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Number of Books', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'genre_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: genre_distribution.png")
        
        # Interpretation
        print("\n📊 Interpretation:")
        print(f"   The dataset contains {genre_counts.iloc[0]} {genre_counts.index[0]} books "
              f"and {genre_counts.iloc[1]} {genre_counts.index[1]} books.")
        
    def _analyze_rating_by_genre(self):
        """Q2: Average user rating per genre"""
        print("\n--- Q2: Average User Rating per Genre ---")
        
        avg_rating = self.df_clean.groupby('genre')['user_rating'].agg(['mean', 'median', 'std'])
        print(avg_rating)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        self.df_clean.boxplot(column='user_rating', by='genre', patch_artist=True)
        plt.suptitle('')
        plt.title('User Rating Distribution by Genre', fontsize=16, fontweight='bold')
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('User Rating', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rating_by_genre.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: rating_by_genre.png")
        
        # Interpretation
        print("\n📊 Interpretation:")
        best_genre = avg_rating['mean'].idxmax()
        best_rating = avg_rating['mean'].max()
        print(f"   {best_genre} books have the highest average rating ({best_rating:.2f}/5.0).")
        
    def _analyze_top_authors(self):
        """Q3: Top 10 authors by number of best sellers"""
        print("\n--- Q3: Top 10 Authors by Number of Best Sellers ---")
        
        top_authors = self.df_clean['author'].value_counts().head(10)
        print(top_authors)
        
        # Visualization
        plt.figure(figsize=(12, 6))
        top_authors.plot(kind='barh', color='#2ecc71')
        plt.title('Top 10 Authors by Number of Best Sellers', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Best Sellers', fontsize=12)
        plt.ylabel('Author', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_authors.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: top_authors.png")
        
        # Interpretation
        print("\n📊 Interpretation:")
        print(f"   {top_authors.index[0]} leads with {top_authors.iloc[0]} best-selling books.")
        
    def _analyze_reviews_ratings_relationship(self):
        """Q4: Relationship between reviews and ratings"""
        print("\n--- Q4: Relationship Between Reviews and Ratings ---")
        
        correlation = self.df_clean['reviews'].corr(self.df_clean['user_rating'])
        print(f"Correlation coefficient: {correlation:.4f}")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df_clean['user_rating'], self.df_clean['reviews'], 
                   alpha=0.6, color='#9b59b6', edgecolors='black', linewidth=0.5)
        plt.title('Relationship Between User Rating and Number of Reviews', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('User Rating', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reviews_vs_ratings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: reviews_vs_ratings.png")
        
        # Interpretation
        print("\n📊 Interpretation:")
        if abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if correlation > 0 else "negative"
        print(f"   There is a {strength} {direction} correlation ({correlation:.4f}) "
              f"between ratings and review counts.")
        
    def _analyze_price_distribution(self):
        """Q5: Price distribution of best-selling books"""
        print("\n--- Q5: Price Distribution of Best-Selling Books ---")
        
        price_stats = self.df_clean['price'].describe()
        print(price_stats)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df_clean['price'], bins=20, color='#f39c12', 
                    edgecolor='black', alpha=0.7)
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Price ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].axvline(price_stats['mean'], color='red', linestyle='--', 
                       label=f'Mean: ${price_stats["mean"]:.2f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(self.df_clean['price'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#f39c12'))
        axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Price ($)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: price_distribution.png")
        
        # Interpretation
        print("\n📊 Interpretation:")
        print(f"   Average book price: ${price_stats['mean']:.2f}")
        print(f"   Price range: ${price_stats['min']:.0f} - ${price_stats['max']:.0f}")
        print(f"   Most books are priced around ${price_stats['50%']:.2f} (median)")
        
    def _analyze_yearly_trends(self):
        """Q6: Year-wise trend of number of best sellers"""
        print("\n--- Q6: Year-wise Trend of Number of Best Sellers ---")
        
        yearly_counts = self.df_clean['year'].value_counts().sort_index()
        print(yearly_counts)
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_counts.index, yearly_counts.values, marker='o', 
                linewidth=2, markersize=8, color='#e67e22')
        plt.title('Year-wise Trend of Best-Selling Books', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Books', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(yearly_counts.index, rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yearly_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: yearly_trends.png")
        
        # Interpretation
        print("\n📊 Interpretation:")
        peak_year = yearly_counts.idxmax()
        peak_count = yearly_counts.max()
        print(f"   Peak year: {peak_year} with {peak_count} best-selling books")
        
    def _analyze_price_rating_correlation(self):
        """Q7: Does higher price correlate with better ratings?"""
        print("\n--- Q7: Price vs Rating Correlation ---")
        
        correlation = self.df_clean['price'].corr(self.df_clean['user_rating'])
        print(f"Correlation coefficient: {correlation:.4f}")
        
        # Visualization with regression line
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df_clean['price'], self.df_clean['user_rating'], 
                   alpha=0.6, color='#1abc9c', edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(self.df_clean['price'], self.df_clean['user_rating'], 1)
        p = np.poly1d(z)
        plt.plot(self.df_clean['price'].sort_values(), 
                p(self.df_clean['price'].sort_values()), 
                "r--", linewidth=2, label=f'Trend line (r={correlation:.3f})')
        
        plt.title('Price vs User Rating Correlation', fontsize=16, fontweight='bold')
        plt.xlabel('Price ($)', fontsize=12)
        plt.ylabel('User Rating', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'price_vs_rating.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: price_vs_rating.png")
        
        # Interpretation
        print("\n📊 Interpretation:")
        if correlation > 0:
            print(f"   Higher prices show a slight positive correlation with ratings ({correlation:.4f}).")
            print("   However, the correlation is weak, suggesting price is not a strong predictor of quality.")
        else:
            print(f"   Price shows a negative correlation with ratings ({correlation:.4f}).")
            print("   This suggests that cheaper books may have slightly better ratings.")
        
    def generate_insights(self):
        """
        Phase 6: Generate key insights and conclusions
        """
        print("\n" + "="*80)
        print("PHASE 6: KEY INSIGHTS & CONCLUSIONS")
        print("="*80)
        
        insights = []
        
        # Insight 1: Genre distribution
        genre_counts = self.df_clean['genre'].value_counts()
        insights.append(f"1. Genre Balance: The dataset shows {genre_counts.iloc[0]} {genre_counts.index[0]} "
                       f"books vs {genre_counts.iloc[1]} {genre_counts.index[1]} books, "
                       f"indicating reading preferences on Amazon's platform.")
        
        # Insight 2: Rating quality
        avg_rating = self.df_clean['user_rating'].mean()
        insights.append(f"2. High Quality Standard: The average rating is {avg_rating:.2f}/5.0, "
                       f"demonstrating that best-selling books maintain high quality standards.")
        
        # Insight 3: Top authors
        top_author = self.df_clean['author'].value_counts().index[0]
        top_count = self.df_clean['author'].value_counts().iloc[0]
        insights.append(f"3. Author Dominance: {top_author} leads with {top_count} best-sellers, "
                       f"showing strong brand recognition and reader loyalty.")
        
        # Insight 4: Price insights
        median_price = self.df_clean['price'].median()
        insights.append(f"4. Affordable Pricing: The median price is ${median_price:.2f}, "
                       f"suggesting best-sellers are priced accessibly to reach wider audiences.")
        
        # Insight 5: Review patterns
        correlation = self.df_clean['reviews'].corr(self.df_clean['user_rating'])
        insights.append(f"5. Reviews vs Ratings: The correlation between review count and rating "
                       f"is {correlation:.3f}, indicating that popular books (more reviews) "
                       f"tend to have {'higher' if correlation > 0 else 'varying'} ratings.")
        
        # Insight 6: Temporal trends
        yearly_counts = self.df_clean['year'].value_counts().sort_index()
        peak_year = yearly_counts.idxmax()
        insights.append(f"6. Temporal Patterns: Year {peak_year} had the most best-sellers, "
                       f"which may reflect market trends or data collection methods.")
        
        # Insight 7: Price-Quality relationship
        price_rating_corr = self.df_clean['price'].corr(self.df_clean['user_rating'])
        insights.append(f"7. Price vs Quality: With a correlation of {price_rating_corr:.3f}, "
                       f"price is not a strong indicator of book quality. "
                       f"Readers value content over cost.")
        
        print("\n📈 KEY BUSINESS INSIGHTS:\n")
        for insight in insights:
            print(f"   {insight}\n")
        
        # Save insights to file
        with open(self.output_dir / 'insights.txt', 'w') as f:
            f.write("AMAZON BEST-SELLING BOOKS - KEY INSIGHTS\n")
            f.write("="*80 + "\n\n")
            for insight in insights:
                f.write(f"{insight}\n\n")
        
        print("✓ Insights saved to: insights.txt")
        
    def run_full_analysis(self):
        """
        Execute the complete analysis pipeline
        """
        print("\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*15 + "AMAZON BEST-SELLING BOOKS ANALYSIS" + " "*29 + "║")
        print("║" + " "*20 + "Complete Data Analysis Pipeline" + " "*26 + "║")
        print("╚" + "="*78 + "╝")
        
        # Execute all phases
        self.load_data()
        self.understand_data()
        self.clean_data()
        self.exploratory_analysis()
        self.generate_insights()
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 All visualizations saved in: {self.output_dir.absolute()}")
        print(f"📊 Total visualizations created: 7")
        print(f"📝 Insights report: {self.output_dir.absolute() / 'insights.txt'}")
        print("\n")


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = AmazonBooksAnalyzer(data_path='data/amazon_books.csv')
    
    # Run complete analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
