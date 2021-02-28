import pandas
pandas.read_json("redis.json").to_excel("output.xlsx")