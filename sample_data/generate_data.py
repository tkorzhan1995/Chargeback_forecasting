"""
Sample data generator for testing the chargeback management system.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
from utils import get_logger

logger = get_logger(__name__)


class SampleDataGenerator:
    """
    Generate synthetic data for testing.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize SampleDataGenerator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
    
    def generate_products(self, n_products: int = 50) -> pd.DataFrame:
        """
        Generate sample products.
        
        Args:
            n_products: Number of products to generate
            
        Returns:
            pd.DataFrame: Products dataframe
        """
        logger.info(f"Generating {n_products} sample products")
        
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys']
        
        products = []
        for i in range(n_products):
            products.append({
                'product_id': f'PROD{i+1:05d}',
                'product_name': f'Product {i+1}',
                'category': random.choice(categories),
                'price': round(random.uniform(10, 500), 2),
            })
        
        return pd.DataFrame(products)
    
    def generate_customers(self, n_customers: int = 1000) -> pd.DataFrame:
        """
        Generate sample customers.
        
        Args:
            n_customers: Number of customers to generate
            
        Returns:
            pd.DataFrame: Customers dataframe
        """
        logger.info(f"Generating {n_customers} sample customers")
        
        segments = ['Premium', 'Regular', 'New']
        
        customers = []
        for i in range(n_customers):
            join_date = datetime.now() - timedelta(days=random.randint(1, 730))
            customers.append({
                'customer_id': f'CUST{i+1:06d}',
                'segment': random.choice(segments),
                'join_date': join_date.strftime('%Y-%m-%d'),
                'total_orders': random.randint(1, 50),
            })
        
        return pd.DataFrame(customers)
    
    def generate_channels(self) -> pd.DataFrame:
        """
        Generate sample channels.
        
        Returns:
            pd.DataFrame: Channels dataframe
        """
        logger.info("Generating sample channels")
        
        channels = [
            {'channel_id': 'CH001', 'channel_name': 'Website', 'channel_type': 'Online'},
            {'channel_id': 'CH002', 'channel_name': 'Mobile App', 'channel_type': 'Online'},
            {'channel_id': 'CH003', 'channel_name': 'In-Store', 'channel_type': 'Physical'},
            {'channel_id': 'CH004', 'channel_name': 'Phone Order', 'channel_type': 'Call Center'},
            {'channel_id': 'CH005', 'channel_name': 'Marketplace', 'channel_type': 'Third Party'},
        ]
        
        return pd.DataFrame(channels)
    
    def generate_transactions(self, n_transactions: int = 10000,
                             products: pd.DataFrame = None,
                             customers: pd.DataFrame = None,
                             channels: pd.DataFrame = None,
                             days_back: int = 180) -> pd.DataFrame:
        """
        Generate sample transactions.
        
        Args:
            n_transactions: Number of transactions to generate
            products: Products dataframe
            customers: Customers dataframe
            channels: Channels dataframe
            days_back: Number of days of historical data
            
        Returns:
            pd.DataFrame: Transactions dataframe
        """
        logger.info(f"Generating {n_transactions} sample transactions")
        
        # Generate reference data if not provided
        if products is None:
            products = self.generate_products()
        if customers is None:
            customers = self.generate_customers()
        if channels is None:
            channels = self.generate_channels()
        
        transactions = []
        start_date = datetime.now() - timedelta(days=days_back)
        
        for i in range(n_transactions):
            product = products.sample(1).iloc[0]
            customer = customers.sample(1).iloc[0]
            channel = channels.sample(1).iloc[0]
            
            txn_date = start_date + timedelta(days=random.randint(0, days_back))
            
            # Add some price variation
            amount = round(product['price'] * random.uniform(0.9, 1.1), 2)
            
            transactions.append({
                'transaction_id': f'TXN{i+1:08d}',
                'transaction_date': txn_date.strftime('%Y-%m-%d %H:%M:%S'),
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'channel_id': channel['channel_id'],
                'amount': amount,
                'currency': 'USD',
                'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer']),
                'status': random.choice(['completed'] * 95 + ['pending', 'cancelled'] * 5),
            })
        
        return pd.DataFrame(transactions)
    
    def generate_chargebacks(self, transactions: pd.DataFrame,
                            chargeback_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate sample chargebacks from transactions.
        
        Args:
            transactions: Transactions dataframe
            chargeback_rate: Percentage of transactions that become chargebacks
            
        Returns:
            pd.DataFrame: Chargebacks dataframe
        """
        logger.info(f"Generating chargebacks with rate {chargeback_rate*100}%")
        
        # Select random transactions to become chargebacks
        n_chargebacks = int(len(transactions) * chargeback_rate)
        chargeback_txns = transactions.sample(n=n_chargebacks)
        
        reason_codes = [
            'Fraud',
            'Product Not Received',
            'Product Not as Described',
            'Duplicate Charge',
            'Credit Not Processed',
            'Subscription Cancelled',
        ]
        
        statuses = ['won', 'lost', 'pending']
        
        chargebacks = []
        for i, (_, txn) in enumerate(chargeback_txns.iterrows()):
            # Chargeback occurs some days after transaction
            txn_date = datetime.strptime(txn['transaction_date'], '%Y-%m-%d %H:%M:%S')
            cb_date = txn_date + timedelta(days=random.randint(5, 60))
            
            chargebacks.append({
                'chargeback_id': f'CB{i+1:08d}',
                'transaction_id': txn['transaction_id'],
                'chargeback_date': cb_date.strftime('%Y-%m-%d %H:%M:%S'),
                'customer_id': txn['customer_id'],
                'product_id': txn['product_id'],
                'channel_id': txn['channel_id'],
                'amount': txn['amount'],
                'reason_code': random.choice(reason_codes),
                'status': random.choice(statuses),
                'currency': txn['currency'],
            })
        
        return pd.DataFrame(chargebacks)
    
    def generate_complete_dataset(self, output_dir: Path = None) -> dict:
        """
        Generate complete sample dataset.
        
        Args:
            output_dir: Directory to save sample data
            
        Returns:
            dict: Dictionary of all generated dataframes
        """
        logger.info("Generating complete sample dataset")
        
        # Generate all data
        products = self.generate_products(50)
        customers = self.generate_customers(1000)
        channels = self.generate_channels()
        transactions = self.generate_transactions(10000, products, customers, channels)
        chargebacks = self.generate_chargebacks(transactions, chargeback_rate=0.02)
        
        datasets = {
            'products': products,
            'customers': customers,
            'channels': channels,
            'transactions': transactions,
            'chargebacks': chargebacks,
        }
        
        # Save to files if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            for name, df in datasets.items():
                filepath = output_dir / f'{name}.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {name} to {filepath}")
        
        logger.info("Sample dataset generation complete")
        return datasets


if __name__ == '__main__':
    # Generate sample data when run directly
    from config.settings import DATA_DIR
    
    generator = SampleDataGenerator()
    datasets = generator.generate_complete_dataset(DATA_DIR)
    
    print("\nGenerated datasets:")
    for name, df in datasets.items():
        print(f"- {name}: {len(df)} records")
