# Databricks notebook source
import pyspark
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('Spam Detection').getOrCreate()

# COMMAND ----------

#df = spark.read.csv('/FileStore/tables/SMSSpamCollectio',inferSchema = True, header = True,sep='\t')
df = sql("SELECT * FROM smsspamcollection_1") #from databreaks table.
df.show()

# COMMAND ----------

from pyspark.sql.functions import length

# COMMAND ----------

df = df.withColumn('length',length(df['text']))

# COMMAND ----------

df.show()

# COMMAND ----------

#avg length of ham and spam messages
df.groupby('class').mean().show()

# COMMAND ----------

from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer,
                                IDF, StringIndexer)

# COMMAND ----------

#data prepareation
tokenizer = Tokenizer(inputCol='Text',outputCol='token_text')
stop_remove = StopWordsRemover(inputCol = 'token_text',outputCol ='stop_token')
count_vec = CountVectorizer(inputCol = 'stop_token',outputCol = 'c_vec')
idf = IDF(inputCol = 'c_vec',outputCol = 'tf_idf')
ham_spam_to_numaric = StringIndexer(inputCol='Class',outputCol='label')


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

# COMMAND ----------

nb = NaiveBayes()

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

#pipeline
data_prep_pipeline = Pipeline(stages=[ham_spam_to_numaric,tokenizer,
                                      stop_remove,count_vec,idf,clean_up])

# COMMAND ----------

cleaner = data_prep_pipeline.fit(df)

# COMMAND ----------

clean_data= cleaner.transform(df)

# COMMAND ----------

cleaner_data.columns

# COMMAND ----------

clean_data = clean_data.select('label','features')

# COMMAND ----------

clean_data.show()

# COMMAND ----------

training,test = clean_data.randomSplit([0.7,0.3])

# COMMAND ----------

spam_detector = nb.fit(training)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

test_results = spam_detector.transform(test)

# COMMAND ----------

test_results.show()

# COMMAND ----------

#label vs prediction by multiclass evaluation metrix
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

acc_evel = MulticlassClassificationEvaluator()

# COMMAND ----------

acc = acc_evel.evaluate(test_results)

# COMMAND ----------

print('ACC of NB Model')
print(acc)

# COMMAND ----------

#From above acc value we can conclude that the accurary of above NB model is 92%.

# COMMAND ----------


