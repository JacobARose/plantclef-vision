"""
Created on Thursday May 1st, 2025
Created by Jacob A Rose

Acollection of commands meant to simplify installing the more unwieldy requirements for plantclef on a new machine, namely
- Java
- Spark

"""


sudo apt-get update
sudo apt install default-jdk

wget https://dlcdn.apache.org/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz
tar -xzf spark-3.5.5-bin-hadoop3.tgz
sudo mv spark-3.5.5-bin-hadoop3 /usr/local/spark


export SPARK_HOME=/usr/local/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
