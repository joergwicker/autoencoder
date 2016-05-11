# Autoencoder

Autoencoder Implementation in Scala, used in and implemented for

```
 Wicker, Jörg; Tyukin, Andrey; Kramer, Stefan

A Nonlinear Label Compression and Transformation Method for Multi-Label Classification using Autoencoders (Inproceeding)
Bailey, James; Khan, Latifur; Washio, Takashi; Dobbie, Gill; Huang, Zhexue Joshua; Wang, Ruili (Ed.): The 20th Pacific Asia Conference on Knowledge Discovery and Data Mining (PAKDD), pp. 328-340, Springer International Publishing, Switzerland, 2016, ISBN: 978-3-319-31753-3.
```

Please cite this paper when using the implementation:

```
@inproceedings{wicker2016nonlinear,
title = {A Nonlinear Label Compression and Transformation Method for Multi-Label Classification using Autoencoders},
author = {Jörg Wicker and Andrey Tyukin and Stefan Kramer},
editor = {James Bailey and Latifur Khan and Takashi Washio and Gill Dobbie and Zhexue Joshua Huang and Ruili Wang},
url = {http://dx.doi.org/10.1007/978-3-319-31753-3_27},
doi = {10.1007/978-3-319-31753-3_27},
isbn = {978-3-319-31753-3},
year = {2016},
date = {2016-04-16},
booktitle = {The 20th Pacific Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
volume = {9651},
pages = {328-340},
publisher = {Springer International Publishing},
address = {Switzerland},
series = {Lecture Notes in Computer Science}
}
```



# Compile

We use maven as a build tool, so simply use

```
mvn clean install
```

to compile the source code, or use this Maven dependency to use it in your programm:

```
    <dependency>
      <groupId>org.kramerlab</groupId>
      <artifactId>autoencoder</artifactId>
      <version>0.1</version>
    </dependency>
``` 

Example usage is in Meka https://github.com/Waikato/meka/blob/master/src/main/java/meka/classifiers/multilabel/Maniac.java#L365