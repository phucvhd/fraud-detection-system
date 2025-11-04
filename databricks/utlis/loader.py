from pyspark.shell import spark


class DataLoader:
    def __init__(self, table_name):
        self.table_name = table_name

    def load(self):
        if self.table_name is None:
            raise ValueError("Table name is required.")
        return spark.sql(f"SELECT * FROM {self.table_name}").toPandas()

# Usage
# loader = DataLoader("workspace.fraud_detection.creditcard")
# df = loader.load()
# display(df)