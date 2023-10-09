### Overview:
A high level overview and workflow as Data Engineers.

### Scope:
Data Engineers Pull data from different sources and organize (structure) them for further use.
### Data Types:
* Structured Data: Relational DataBases.
* Semi Structured Data: XML, CSV, JSON
* Unstructured Data: PDFs, Emails, Other Doc format
* Binary Data: Media files || its harder to organize or categorized 

Note: kaggle.com is good source to get free data sets

### Terms:
- Data mining: Collection of data from different source (From Pipeline)
- Big: lots of data (more data that can be handled by a single personal computers). Usually these data are handled by cloud computers (clusters).|| AWS, Azure, Google Cloud || Hadoop and NoSql are being used to handle Big Data.
- Data Pipeline: It's a kind pipeline Data Engineers Build to pull/collect data flow form different DataSource (and build data lakes). 

- Data Lakes: It's a kind of a storage of un-structured/un-organized data that come from the Data Source Streams.
- Data Warehouse: From Data Lakes Data Engineers Extract Meaningful Data, organize and store (formation of Big Data) them in a way that other company/organizations can use them. This is the warehouse.

### Tools:
* Kafka : helps to collect and create the data lake form stream (different source) of data
* Azure | Hadoop | Amazon S3 : to hold/contain the data lakes
* Amazon Athena/Redshift | Google BigQuery : 

### What Data Engineers Do:
1. Build "Extract Transform Load" (ETL) Pipeline
2. Build Analysis Tools and system monitoring tools
3. Maintain the Data lake and the warehouse

### Types of DB:

* Relational DB: PostGres, MySQL (Good for single DB)

* NoSQL / Non Relational DB: Good for Distributed DB or Multiple DB working together. MongoDB, Redis, Cassandra, CouchDB

* NewSQL: Combination of Relational and Scalable DB structure of NoSQl. ClustrixDB, NuoDB, CockroachDB, Pivotal GemFire XD, Altibase, MemSQL, VoltDB, c-treeACE, Percona TokuDB, Apache Trafodion, TIBCO ActiveSpaces, ActorDB

* Search DB : Optimized for searching. Elastic-Search, Solar

* Computational DB: Optimized for computation. Apache-Spark.

* OLTP: online transaction processing DB, like MySQL 

* OLAP: Online analytical processing DB. Can be used to analyze business data from different points of view. OLAP is optimized for complex data analysis and reporting. these databases are created in Analysis Server instances only. Cognos Powerplay, Oracle Database OLAP Option, MicroStrategy, Microsoft Analysis Services, Essbase, TM1, Jedox, and icCube.

* DBMS: Database Management System. Its a combination of tools (In a perticular DB System) that are used to control (Commit CRUD Operations, indexing, etc) 

### Hadoop:
It's a data lake solution. Created by Yahoo and later donated to Apache. It can hold/store and manage huge amount of data within clusters of computer.
- HDFS : Hadoop Distributed File System (allow store/maintain files on several computers). It's scalable
- MapReduce : help to batch process data
- HIVE : Alternate of MapReduce to batch process data on Hadoop

### Apache Spark (Batching Processor):
- in-memory-process : much faster than MapReduce

### Apache Flink (Real Time / Stream Processing):
- Process data on the Fly: Data is processed while streamed, so no precessing over head or batch processing after.

### Apache Kafka (Stream/Real-time Processing):
Apache Kafka is an open-source distributed event streaming platform. It can receive raw data from different data sources, then precess and/or pass different locations/clusters.