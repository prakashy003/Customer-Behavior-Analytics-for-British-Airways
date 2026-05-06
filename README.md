# Customer Behavior Analytics — British Airways

A two-part data science project completed as part of the British Airways Data Science Job Simulation. The project covers end-to-end NLP analysis of customer reviews and a supervised machine learning model to predict booking completion.

---

## Project Structure

```
├── Web scraping to gain company insights/
│   ├── Data-collection(1).ipynb      # Web scraping pipeline
│   ├── Data Cleaning(2).ipynb        # Text preprocessing & cleaning
│   ├── EDA(3).ipynb                  # Exploratory analysis & NLP
│   ├── BA_reviews.csv                # Raw scraped reviews
│   └── cleaned-BA-reviews.csv        # Cleaned & feature-engineered reviews
│
└── Predicting customer buying behaviour/
    ├── Predictive_model.ipynb        # Feature engineering & model training
    └── customer_booking.csv          # Booking dataset
```

---

## Task 1 — Web Scraping & Customer Review Analysis

### Objective
Scrape British Airways customer reviews from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways), clean the data, and extract actionable insights using NLP techniques.

### Data Collection
- Scraped **3,418 reviews** across 35 pages using `requests` and `BeautifulSoup`
- Features collected: review text, star rating (1–10), review date, reviewer country

### Data Cleaning
- Extracted a `verified` flag (Trip Verified vs. Not Verified)
- Removed punctuation, lowercased text, applied **lemmatization** and stopword removal via `nltk`
- Cleaned star ratings (stripped whitespace/escape characters), dropped 5 rows with `None` ratings
- Parsed dates to `datetime` format
- Final clean dataset: **3,411 rows × 6 columns** from **69 unique countries**

### Exploratory Data Analysis

| Metric | Value |
|---|---|
| Average rating | 4.84 / 10 |
| Total reviews | 3,411 |
| Countries represented | 69 |
| Most common rating | 1-star (21.55% of reviews) |

Key findings:
- **Rating distribution** skews heavily negative — 1-star reviews account for over a fifth of all reviews
- **N-gram analysis** of high-rated reviews (7–10 stars) reveals positive associations with cabin crew: *"cabin crew friendly helpful"*, *"cabin crew friendly attentive"*
- **Word frequency analysis** shows seat, service, and food are the most discussed topics across all ratings
- **Time-series** review volume dropped significantly between April 2020 and August 2021, consistent with COVID-19 travel restrictions

### Sentiment Analysis
- **TextBlob** polarity scores: 2,286 reviews fell in the neutral band (−0.2 to 0.2)
- **VADER** compound score (threshold ±0.2):

  | Sentiment | Count |
  |---|---|
  | Positive | 2,245 |
  | Negative | 1,049 |
  | Neutral | 117 |

### Topic Modeling
- **LDA** (8 topics): identified themes around seat/class experience, cabin crew & meals, delay/baggage issues, and lounge experience
- **NMF** (2 topics): separated in-flight service topics from operational/logistics topics

### Libraries
`pandas` · `numpy` · `requests` · `BeautifulSoup` · `matplotlib` · `seaborn` · `plotly` · `wordcloud` · `nltk` · `textblob` · `scikit-learn`

---

## Task 2 — Predicting Customer Buying Behaviour

### Objective
Build a machine learning model to predict whether a customer will complete a flight booking, and identify the features that most influence purchase decisions.

### Dataset
- Source: `customer_booking.csv` — anonymized booking records
- Features include: `sales_channel`, `trip_type`, `booking_origin`, `route`, flight details, and ancillary service flags
- Target: `booking_complete` (binary: 0 = not completed, 1 = completed)

### Feature Engineering & Preprocessing
- One-hot encoded categorical columns: `sales_channel`, `trip_type`, `booking_origin`, `route`
- Applied `StandardScaler` normalization to all numeric features
- Addressed **class imbalance** by downsampling the majority class (not-completed) to 8,000 samples, balancing it against the minority class

### Model — Random Forest Classifier

| Parameter | Value |
|---|---|
| `n_estimators` | 50 |
| `max_depth` | 50 |
| `min_samples_split` | 5 |
| Train / Test split | 80% / 20% |
| Random state | 42 |

### Evaluation
- Evaluated on accuracy, precision, F1-score, and confusion matrix (via `yellowbrick`)
- Feature importance plot identifies which booking attributes most strongly predict purchase completion
- Low initial F1-score on imbalanced data was improved after downsampling the majority class

### Libraries
`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `yellowbrick`

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd Customer-Behavior-Analytics-for-British-Airways

# Install dependencies
pip install pandas numpy requests beautifulsoup4 matplotlib seaborn plotly \
            wordcloud nltk textblob scikit-learn yellowbrick
```

Download required NLTK assets before running the NLP notebooks:

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

Run the notebooks in order within each task folder:
1. `Data-collection(1).ipynb`
2. `Data Cleaning(2).ipynb`
3. `EDA(3).ipynb`

---

## Key Takeaways

- British Airways customers rate their experience below average on Skytrax, with negative reviews dominating
- Cabin crew service is the most consistently praised aspect among high-rated reviews
- Seat comfort, food quality, and operational issues (delays, baggage) are the primary pain points
- Booking completion is predictable from booking metadata — route, lead time, and ancillary service selections carry the highest feature importance

---

## Project Extensions

The original Forage simulation laid the groundwork. This project is being actively extended beyond the brief with the following additions:

### Phase 1 — Airline Industry Competitor Benchmarking *(in progress)*
Extending the scraping and NLP pipeline to cover **Emirates, Singapore Airlines, and Qatar Airways** alongside British Airways — four of the most reviewed airlines globally on Skytrax. The goal is to move from a single-airline analysis to a comparative industry view: where does BA genuinely underperform, which complaints are industry-wide, and which are BA-specific? Output will be a unified dataset and a comparative analysis notebook covering rating distributions, per-airline sentiment scores, and top complaint themes by carrier.

### Phase 2 — Interactive Multi-Airline Dashboard (Streamlit)
A publicly hosted Streamlit web application giving an interactive, side-by-side view of all four airlines. Planned tabs: overall ranking comparison, per-airline sentiment trend over time, NLP topic explorer by rating band, and country-level review heatmap. Will be deployed on Streamlit Cloud with a live URL linked here upon completion.

### Phase 3 — Business Intelligence Report (Power BI)
A Power BI report built on the final merged dataset — designed to mirror what an internal analyst would present to a leadership team. Covers KPI cards (avg rating, % positive reviews, review volume), cross-airline rating comparison, and a geographic breakdown of reviewer sentiment. The `.pbix` file and exported report screenshots will be committed to the repository.

---

## Acknowledgements

This project was completed as part of the **British Airways Data Science Virtual Experience Programme** on Forage, and subsequently extended independently to build a broader airline industry analytics platform.
