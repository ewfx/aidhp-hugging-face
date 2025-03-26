

!pip install pyspark
!pip install faker
!pip install flask
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, explode, lit, when, count, desc
from faker import Faker
import random
import json



spark = SparkSession.builder.appName("HyperPersonalizedRecommendations").getOrCreate()

fake = Faker()

products = [
    ("P1001", "Apple iPhone 15 Pro Max", "Electronics", 1099.99, 0.95),
    ("P1002", "Samsung Galaxy S23 Ultra", "Electronics", 1199.99, 0.92),
    ("P1003", "Sony WH-1000XM5 Headphones", "Electronics", 399.99, 0.90),
    ("P1004", "Nike Air Zoom Pegasus 40", "Fashion", 129.99, 0.88),
    ("P1005", "Adidas Ultraboost Light", "Fashion", 189.99, 0.85),
    ("P1006", "The North Face Thermoball Eco Jacket", "Fashion", 229.99, 0.82),
    ("P1007", "Instant Pot Duo 7-in-1", "Home", 99.99, 0.89),
    ("P1008", "Dyson V15 Detect Vacuum", "Home", 699.99, 0.91),
    ("P1009", "Breville Barista Express", "Home", 699.99, 0.87),
    ("P1010", "Kindle Paperwhite Signature", "Books", 189.99, 0.84),
    ("P1011", "Apple AirTag 4-Pack", "Electronics", 99.99, 0.83),
    ("P1012", "Amazon Echo Dot (5th Gen)", "Electronics", 49.99, 0.80),
    ("P1013", "Yeti Rambler 20oz Tumbler", "Home", 39.99, 0.78),
    ("P1014", "Patagonia Nano Puff Jacket", "Fashion", 229.99, 0.86),
    ("P1015", "Allbirds Wool Runners", "Fashion", 115.00, 0.81)
]


num_customers = 1000
customer_data = [(i, fake.name(), fake.random_int(min=18, max=70), fake.city(),
                  fake.random_int(min=20000, max=150000), random.choice(["Male", "Female"]))
                 for i in range(1, num_customers+1)]

customer_df = spark.createDataFrame(customer_data, ["customer_id", "name", "age", "city", "income", "gender"])


social_posts = [
    ("Just bought a new iPhone! Love the camera quality.", "Electronics"),
    ("Looking for the best travel destinations for summer!", "Travel"),
    ("Thinking about investing in tech stocks this quarter.", "Finance"),
    ("Love shopping for new fashion trends this season!", "Fashion"),
    ("Need a good cashback credit card for daily purchases.", "Credit Card"),
]

social_data = [(random.randint(1, num_customers), random.choice(social_posts)[0], random.choice(social_posts)[1])
               for _ in range(2000)]

social_df = spark.createDataFrame(social_data, ["customer_id", "post", "category"])


products_df = spark.createDataFrame(products, ["product_id", "product_name", "category", "price", "confidence_score"])

transaction_data = []
for _ in range(5000):
    customer_id = random.randint(1, num_customers)
    product = random.choice(products)
    transaction_data.append((
        customer_id,
        product[0],
        product[1],
        product[2],
        product[3],
        product[4],
        random.randint(1, 3)
    ))

transaction_df = spark.createDataFrame(transaction_data,
                                     ["customer_id", "product_id", "product_name", "category", "price", "confidence_score", "quantity"])


demographics_data = [(i, random.choice(["Single", "Married", "Divorced"]),
                      random.choice(["Urban", "Suburban", "Rural"]),
                      random.choice(["Tech-savvy", "Traditional", "Luxury Buyer", "Value Seeker", "Early Adopter"]))
                     for i in range(1, num_customers+1)]

demographics_df = spark.createDataFrame(demographics_data,
                                      ["customer_id", "marital_status", "residence_type", "buyer_persona"])


from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import expr


indexer = StringIndexer(inputCol="category", outputCol="category_index")
indexed_transaction_df = indexer.fit(transaction_df).transform(transaction_df)


indexed_transaction_df = indexed_transaction_df.withColumn(
    "rating",
    expr("confidence_score * quantity")  )


als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="customer_id",
    itemCol="category_index",
    ratingCol="rating",
    coldStartStrategy="drop"
)
model = als.fit(indexed_transaction_df)


category_recs = model.recommendForAllUsers(1)


category_labels = indexer.fit(transaction_df).labels
category_map = {i: category_labels[i] for i in range(len(category_labels))}


def get_best_recommendation(customer_id):

    if customer_id < 1 or customer_id > num_customers:
        return json.dumps({"error": "Customer ID not found"}, separators=(',', ':'))


    customer = customer_df.filter(col("customer_id") == customer_id).first()
    demo = demographics_df.filter(col("customer_id") == customer_id).first()


    recs = category_recs.filter(col("customer_id") == customer_id).first()

    if not recs:
        return json.dumps({"error": "No recommendations available for this customer"}, separators=(',', ':'))


    category_idx = recs["recommendations"][0]["category_index"]
    category_name = category_map.get(category_idx, "Unknown")


    best_product = (transaction_df.filter(col("category") == category_name)
                   .orderBy(desc("confidence_score"))
                   .select("product_id", "product_name", "confidence_score")
                   .first())

    if not best_product:
        return json.dumps({"error": "No products found in recommended category"}, separators=(',', ':'))


    recent_purchase = (transaction_df.filter(col("customer_id") == customer_id)
                      .orderBy(desc("confidence_score"))
                      .select("product_name", "category")
                      .first())


    output = {
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "recommendation": {
            "product_name": best_product["product_name"],
            "category": category_name
        }
    }

    return json.dumps(output, separators=(',', ':'))

sample_customer = random.randint(1, num_customers)
print(get_best_recommendation(sample_customer))
