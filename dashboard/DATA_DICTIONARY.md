# Data Dictionary

## Transactions Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| transaction_id | string | Unique transaction identifier | TXN00000001 |
| transaction_date | datetime | Date and time of transaction | 2026-01-15 14:23:45 |
| customer_id | string | Customer identifier | CUST000001 |
| product_id | string | Product identifier | PROD00001 |
| channel_id | string | Sales channel identifier | CH001 |
| amount | float | Transaction amount | 125.50 |
| currency | string | Currency code | USD |
| payment_method | string | Payment method used | Credit Card |
| status | string | Transaction status | completed, pending, cancelled |

## Chargebacks Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| chargeback_id | string | Unique chargeback identifier | CB00000001 |
| transaction_id | string | Related transaction ID | TXN00000001 |
| chargeback_date | datetime | Date of chargeback | 2026-01-25 10:15:30 |
| customer_id | string | Customer identifier | CUST000001 |
| product_id | string | Product identifier | PROD00001 |
| channel_id | string | Sales channel identifier | CH001 |
| amount | float | Chargeback amount | 125.50 |
| reason_code | string | Reason for chargeback | Fraud, Product Not Received |
| status | string | Chargeback status | won, lost, pending |
| currency | string | Currency code | USD |

## Products Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| product_id | string | Unique product identifier | PROD00001 |
| product_name | string | Product name | Product 1 |
| category | string | Product category | Electronics, Clothing |
| price | float | Standard product price | 125.50 |

## Customers Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| customer_id | string | Unique customer identifier | CUST000001 |
| segment | string | Customer segment | Premium, Regular, New |
| join_date | date | Customer registration date | 2025-06-15 |
| total_orders | integer | Lifetime order count | 25 |

## Channels Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| channel_id | string | Unique channel identifier | CH001 |
| channel_name | string | Channel name | Website, Mobile App |
| channel_type | string | Channel type | Online, Physical, Third Party |

## Reconciliation Mapping Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| chargeback_id | string | Chargeback identifier | CB00000001 |
| transaction_id | string | Matched transaction ID | TXN00000001 |
| match_confidence | float | Confidence score (0-1) | 0.95 |
| match_method | string | Matching method used | exact, fuzzy, amount_time |
| matched_date | datetime | Date of matching | 2026-01-26 08:30:00 |

## Predictions Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| date | date | Prediction date | 2026-02-01 |
| model | string | Model used | ARIMA, Prophet, Ensemble |
| predicted_volume | float | Predicted chargeback count | 45.2 |
| lower_bound | float | Lower confidence bound | 38.5 |
| upper_bound | float | Upper confidence bound | 52.0 |
| confidence_level | float | Confidence level | 0.95 |

## Risk Scores Table

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| transaction_id | string | Transaction identifier | TXN00000001 |
| chargeback_risk_score | float | Risk probability (0-1) | 0.72 |
| high_risk_flag | boolean | High risk indicator | 1 (yes), 0 (no) |
| risk_category | string | Risk category | Low, Medium, High |
| expected_chargeback_amount | float | Expected chargeback amount | 90.25 |

## Data Relationships

```
transactions (1) ← (M) chargebacks
    ├── transaction_id → transaction_id

transactions (M) → (1) products
    ├── product_id → product_id

transactions (M) → (1) customers
    ├── customer_id → customer_id

transactions (M) → (1) channels
    ├── channel_id → channel_id

chargebacks (1) ← (1) reconciliation_mapping
    ├── chargeback_id → chargeback_id

transactions (1) ← (1) reconciliation_mapping
    ├── transaction_id → transaction_id
```

## Calculated Fields

### Chargeback Rate
```
chargeback_rate = (chargeback_count / transaction_count) * 100
```

### Win/Loss Ratio
```
win_loss_ratio = won_chargebacks / lost_chargebacks
win_rate = (won_chargebacks / total_resolved_chargebacks) * 100
```

### Match Rate
```
match_rate = (matched_chargebacks / total_chargebacks) * 100
```

### Average Chargeback Amount
```
avg_chargeback_amount = total_chargeback_amount / chargeback_count
```

## Data Quality Metrics

| Metric | Description | Acceptable Range |
|--------|-------------|------------------|
| Completeness | % of non-null values | > 95% |
| Uniqueness | % of unique IDs | 100% for ID fields |
| Validity | % of values in valid ranges | > 99% |
| Timeliness | Data age | < 24 hours |
| Accuracy | % matching external source | > 98% |

## Date Formats

- **Standard Format**: YYYY-MM-DD HH:MM:SS (2026-01-15 14:23:45)
- **Date Only**: YYYY-MM-DD (2026-01-15)
- **Timezone**: UTC by default

## Amount Formats

- **Currency**: Two decimal places (125.50)
- **Large Numbers**: Use thousands separator in reports (1,250.50)
- **Negative Values**: Indicate refunds or reversals

## Status Values

### Transaction Status
- `completed`: Transaction successfully completed
- `pending`: Transaction awaiting completion
- `cancelled`: Transaction cancelled
- `refunded`: Transaction refunded

### Chargeback Status
- `won`: Chargeback won by merchant
- `lost`: Chargeback lost by merchant
- `pending`: Chargeback under review
- `expired`: Chargeback expired without resolution
- `accepted`: Chargeback accepted without dispute

### Match Method
- `exact`: Exact match on transaction ID
- `fuzzy`: Fuzzy string matching
- `amount_time`: Amount and time-based matching
- `customer_amount`: Customer and amount matching
- `partial_amount`: Partial amount matching
- `probabilistic`: Probabilistic linkage

## Aggregation Periods

- `D`: Daily
- `W`: Weekly  
- `M`: Monthly
- `Q`: Quarterly
- `Y`: Yearly
