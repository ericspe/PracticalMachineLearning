### Practical Machine Learning: Week 4 Prediction Assignment

Summary
-------

"Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes."

From the source: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>

The measures come from weareable devices and this project is about predicting the Class (A,B,C,D,E) based on the device outputs for a sample test data set as provided

Initialize the training and testing data sets
---------------------------------------------

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
set.seed(3433)

#read the training data
urlTraining <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(urlTraining, destfile="pml-training.csv")
dataTrain <- read.csv("pml-training.csv", header=TRUE)

#read the testing data
urlTesting <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(urlTesting, destfile="pml-testing.csv")
dataTest <- read.csv("pml-testing.csv", header=TRUE)
```

Data Exploration
----------------

``` r
dim(dataTrain)
```

    ## [1] 19622   160

``` r
dim(dataTest)
```

    ## [1]  20 160

The number of columns matches (160) in both dataset Lets check whether they are the same:

-   In Training set and not in Testing set:

``` r
setdiff(names(dataTrain), names(dataTest))
```

    ## [1] "classe"

-   In Testing set and not in Trainig Set:

``` r
setdiff(names(dataTest), names(dataTrain))
```

    ## [1] "problem_id"

The predicted column "Classe" is only in the training set. The column "problem\_id" doesn't exist is the testing set.

Lets take a closer look into the data:

``` r
head(dataTrain)
```

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
    ## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
    ## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
    ## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
    ## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
    ##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
    ## 1         no         11      1.41       8.07    -94.4                3
    ## 2         no         11      1.41       8.07    -94.4                3
    ## 3         no         11      1.42       8.07    -94.4                3
    ## 4         no         12      1.48       8.05    -94.4                3
    ## 5         no         12      1.48       8.07    -94.4                3
    ## 6         no         12      1.45       8.06    -94.4                3
    ##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
    ## 1                                                         
    ## 2                                                         
    ## 3                                                         
    ## 4                                                         
    ## 5                                                         
    ## 6                                                         
    ##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
    ## 1                                                                      NA
    ## 2                                                                      NA
    ## 3                                                                      NA
    ## 4                                                                      NA
    ## 5                                                                      NA
    ## 6                                                                      NA
    ##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
    ## 1             NA                         NA             NA             
    ## 2             NA                         NA             NA             
    ## 3             NA                         NA             NA             
    ## 4             NA                         NA             NA             
    ## 5             NA                         NA             NA             
    ## 6             NA                         NA             NA             
    ##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
    ## 1                  NA                   NA                   
    ## 2                  NA                   NA                   
    ## 3                  NA                   NA                   
    ## 4                  NA                   NA                   
    ## 5                  NA                   NA                   
    ## 6                  NA                   NA                   
    ##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
    ## 1                   NA            NA               NA            NA
    ## 2                   NA            NA               NA            NA
    ## 3                   NA            NA               NA            NA
    ## 4                   NA            NA               NA            NA
    ## 5                   NA            NA               NA            NA
    ## 6                   NA            NA               NA            NA
    ##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
    ## 1             NA                NA             NA           NA
    ## 2             NA                NA             NA           NA
    ## 3             NA                NA             NA           NA
    ## 4             NA                NA             NA           NA
    ## 5             NA                NA             NA           NA
    ## 6             NA                NA             NA           NA
    ##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
    ## 1              NA           NA         0.00         0.00        -0.02
    ## 2              NA           NA         0.02         0.00        -0.02
    ## 3              NA           NA         0.00         0.00        -0.02
    ## 4              NA           NA         0.02         0.00        -0.03
    ## 5              NA           NA         0.02         0.02        -0.02
    ## 6              NA           NA         0.02         0.00        -0.02
    ##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
    ## 1          -21            4           22            -3           599
    ## 2          -22            4           22            -7           608
    ## 3          -20            5           23            -2           600
    ## 4          -22            3           21            -6           604
    ## 5          -21            2           24            -6           600
    ## 6          -21            4           21             0           603
    ##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
    ## 1          -313     -128      22.5    -161              34            NA
    ## 2          -311     -128      22.5    -161              34            NA
    ## 3          -305     -128      22.5    -161              34            NA
    ## 4          -310     -128      22.1    -161              34            NA
    ## 5          -302     -128      22.1    -161              34            NA
    ## 6          -312     -128      22.0    -161              34            NA
    ##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
    ## 1           NA              NA           NA            NA               NA
    ## 2           NA              NA           NA            NA               NA
    ## 3           NA              NA           NA            NA               NA
    ## 4           NA              NA           NA            NA               NA
    ## 5           NA              NA           NA            NA               NA
    ## 6           NA              NA           NA            NA               NA
    ##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
    ## 1            NA          NA             NA          NA        0.00
    ## 2            NA          NA             NA          NA        0.02
    ## 3            NA          NA             NA          NA        0.02
    ## 4            NA          NA             NA          NA        0.02
    ## 5            NA          NA             NA          NA        0.00
    ## 6            NA          NA             NA          NA        0.02
    ##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
    ## 1        0.00       -0.02        -288         109        -123         -368
    ## 2       -0.02       -0.02        -290         110        -125         -369
    ## 3       -0.02       -0.02        -289         110        -126         -368
    ## 4       -0.03        0.02        -289         111        -123         -372
    ## 5       -0.03        0.00        -289         111        -123         -374
    ## 6       -0.03        0.00        -289         111        -122         -369
    ##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
    ## 1          337          516                                     
    ## 2          337          513                                     
    ## 3          344          513                                     
    ## 4          344          512                                     
    ## 5          337          506                                     
    ## 6          342          513                                     
    ##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
    ## 1                                                                       
    ## 2                                                                       
    ## 3                                                                       
    ## 4                                                                       
    ## 5                                                                       
    ## 6                                                                       
    ##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
    ## 1           NA            NA          NA           NA            NA
    ## 2           NA            NA          NA           NA            NA
    ## 3           NA            NA          NA           NA            NA
    ## 4           NA            NA          NA           NA            NA
    ## 5           NA            NA          NA           NA            NA
    ## 6           NA            NA          NA           NA            NA
    ##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
    ## 1          NA                 NA                  NA                NA
    ## 2          NA                 NA                  NA                NA
    ## 3          NA                 NA                  NA                NA
    ## 4          NA                 NA                  NA                NA
    ## 5          NA                 NA                  NA                NA
    ## 6          NA                 NA                  NA                NA
    ##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
    ## 1      13.05217      -70.49400    -84.87394                       
    ## 2      13.13074      -70.63751    -84.71065                       
    ## 3      12.85075      -70.27812    -85.14078                       
    ## 4      13.43120      -70.39379    -84.87363                       
    ## 5      13.37872      -70.42856    -84.85306                       
    ## 6      13.38246      -70.81759    -84.46500                       
    ##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
    ## 1                                                                     
    ## 2                                                                     
    ## 3                                                                     
    ## 4                                                                     
    ## 5                                                                     
    ## 6                                                                     
    ##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
    ## 1                                                              NA
    ## 2                                                              NA
    ## 3                                                              NA
    ## 4                                                              NA
    ## 5                                                              NA
    ## 6                                                              NA
    ##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
    ## 1                 NA                                 NA                 NA
    ## 2                 NA                                 NA                 NA
    ## 3                 NA                                 NA                 NA
    ## 4                 NA                                 NA                 NA
    ## 5                 NA                                 NA                 NA
    ## 6                 NA                                 NA                 NA
    ##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
    ## 1                                       NA                       NA
    ## 2                                       NA                       NA
    ## 3                                       NA                       NA
    ## 4                                       NA                       NA
    ## 5                                       NA                       NA
    ## 6                                       NA                       NA
    ##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
    ## 1                                          37                 NA
    ## 2                                          37                 NA
    ## 3                                          37                 NA
    ## 4                                          37                 NA
    ## 5                                          37                 NA
    ## 6                                          37                 NA
    ##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
    ## 1                NA                   NA                NA
    ## 2                NA                   NA                NA
    ## 3                NA                   NA                NA
    ## 4                NA                   NA                NA
    ## 5                NA                   NA                NA
    ## 6                NA                   NA                NA
    ##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
    ## 1                 NA                    NA                 NA
    ## 2                 NA                    NA                 NA
    ## 3                 NA                    NA                 NA
    ## 4                 NA                    NA                 NA
    ## 5                 NA                    NA                 NA
    ## 6                 NA                    NA                 NA
    ##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
    ## 1               NA                  NA               NA                0
    ## 2               NA                  NA               NA                0
    ## 3               NA                  NA               NA                0
    ## 4               NA                  NA               NA                0
    ## 5               NA                  NA               NA                0
    ## 6               NA                  NA               NA                0
    ##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
    ## 1            -0.02             0.00             -234               47
    ## 2            -0.02             0.00             -233               47
    ## 3            -0.02             0.00             -232               46
    ## 4            -0.02            -0.02             -232               48
    ## 5            -0.02             0.00             -233               48
    ## 6            -0.02             0.00             -234               48
    ##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
    ## 1             -271              -559               293               -65
    ## 2             -269              -555               296               -64
    ## 3             -270              -561               298               -63
    ## 4             -269              -552               303               -60
    ## 5             -270              -554               292               -68
    ## 6             -269              -558               294               -66
    ##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
    ## 1         28.4         -63.9        -153                      
    ## 2         28.3         -63.9        -153                      
    ## 3         28.3         -63.9        -152                      
    ## 4         28.1         -63.9        -152                      
    ## 5         28.0         -63.9        -152                      
    ## 6         27.9         -63.9        -152                      
    ##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
    ## 1                                                                  
    ## 2                                                                  
    ## 3                                                                  
    ## 4                                                                  
    ## 5                                                                  
    ## 6                                                                  
    ##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
    ## 1                                                           NA
    ## 2                                                           NA
    ## 3                                                           NA
    ## 4                                                           NA
    ## 5                                                           NA
    ## 6                                                           NA
    ##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
    ## 1                NA                               NA                NA
    ## 2                NA                               NA                NA
    ## 3                NA                               NA                NA
    ## 4                NA                               NA                NA
    ## 5                NA                               NA                NA
    ## 6                NA                               NA                NA
    ##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
    ## 1                                     NA                      NA
    ## 2                                     NA                      NA
    ## 3                                     NA                      NA
    ## 4                                     NA                      NA
    ## 5                                     NA                      NA
    ## 6                                     NA                      NA
    ##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
    ## 1                                        36                NA
    ## 2                                        36                NA
    ## 3                                        36                NA
    ## 4                                        36                NA
    ## 5                                        36                NA
    ## 6                                        36                NA
    ##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
    ## 1               NA                  NA               NA                NA
    ## 2               NA                  NA               NA                NA
    ## 3               NA                  NA               NA                NA
    ## 4               NA                  NA               NA                NA
    ## 5               NA                  NA               NA                NA
    ## 6               NA                  NA               NA                NA
    ##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
    ## 1                   NA                NA              NA
    ## 2                   NA                NA              NA
    ## 3                   NA                NA              NA
    ## 4                   NA                NA              NA
    ## 5                   NA                NA              NA
    ## 6                   NA                NA              NA
    ##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
    ## 1                 NA              NA            0.03            0.00
    ## 2                 NA              NA            0.02            0.00
    ## 3                 NA              NA            0.03           -0.02
    ## 4                 NA              NA            0.02           -0.02
    ## 5                 NA              NA            0.02            0.00
    ## 6                 NA              NA            0.02           -0.02
    ##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
    ## 1           -0.02             192             203            -215
    ## 2           -0.02             192             203            -216
    ## 3            0.00             196             204            -213
    ## 4            0.00             189             206            -214
    ## 5           -0.02             189             206            -214
    ## 6           -0.03             193             203            -215
    ##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
    ## 1              -17              654              476      A
    ## 2              -18              661              473      A
    ## 3              -18              658              469      A
    ## 4              -16              658              469      A
    ## 5              -17              655              473      A
    ## 6               -9              660              478      A

Several columns seem to contain mainly no usable values (NA). The first columns seem not to contain any information about the measures. Otherwise the data is mainly numeric as expected since these are outpouts of weareable devices:

``` r
sapply(dataTrain, typeof)
```

    ##                        X                user_name     raw_timestamp_part_1 
    ##                "integer"                "integer"                "integer" 
    ##     raw_timestamp_part_2           cvtd_timestamp               new_window 
    ##                "integer"                "integer"                "integer" 
    ##               num_window                roll_belt               pitch_belt 
    ##                "integer"                 "double"                 "double" 
    ##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
    ##                 "double"                "integer"                "integer" 
    ##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
    ##                "integer"                "integer"                "integer" 
    ##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
    ##                "integer"                "integer"                 "double" 
    ##           max_picth_belt             max_yaw_belt            min_roll_belt 
    ##                "integer"                "integer"                 "double" 
    ##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
    ##                "integer"                "integer"                 "double" 
    ##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
    ##                "integer"                "integer"                 "double" 
    ##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
    ##                 "double"                 "double"                 "double" 
    ##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
    ##                 "double"                 "double"                 "double" 
    ##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
    ##                 "double"                 "double"                 "double" 
    ##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
    ##                 "double"                 "double"                 "double" 
    ##             accel_belt_x             accel_belt_y             accel_belt_z 
    ##                "integer"                "integer"                "integer" 
    ##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
    ##                "integer"                "integer"                "integer" 
    ##                 roll_arm                pitch_arm                  yaw_arm 
    ##                 "double"                 "double"                 "double" 
    ##          total_accel_arm            var_accel_arm             avg_roll_arm 
    ##                "integer"                 "double"                 "double" 
    ##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
    ##                 "double"                 "double"                 "double" 
    ##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
    ##                 "double"                 "double"                 "double" 
    ##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
    ##                 "double"                 "double"                 "double" 
    ##              gyros_arm_y              gyros_arm_z              accel_arm_x 
    ##                 "double"                 "double"                "integer" 
    ##              accel_arm_y              accel_arm_z             magnet_arm_x 
    ##                "integer"                "integer"                "integer" 
    ##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
    ##                "integer"                "integer"                "integer" 
    ##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
    ##                "integer"                "integer"                "integer" 
    ##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
    ##                "integer"                "integer"                 "double" 
    ##            max_picth_arm              max_yaw_arm             min_roll_arm 
    ##                 "double"                "integer"                 "double" 
    ##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
    ##                 "double"                "integer"                 "double" 
    ##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
    ##                 "double"                "integer"                 "double" 
    ##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
    ##                 "double"                 "double"                "integer" 
    ##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
    ##                "integer"                "integer"                "integer" 
    ##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
    ##                "integer"                "integer"                 "double" 
    ##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
    ##                 "double"                "integer"                 "double" 
    ##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
    ##                 "double"                "integer"                 "double" 
    ## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
    ##                 "double"                "integer"                "integer" 
    ##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
    ##                 "double"                 "double"                 "double" 
    ##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
    ##                 "double"                 "double"                 "double" 
    ##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
    ##                 "double"                 "double"                 "double" 
    ##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
    ##                 "double"                 "double"                 "double" 
    ##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
    ##                 "double"                "integer"                "integer" 
    ##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
    ##                "integer"                "integer"                "integer" 
    ##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
    ##                 "double"                 "double"                 "double" 
    ##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
    ##                 "double"                "integer"                "integer" 
    ##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
    ##                "integer"                "integer"                "integer" 
    ##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
    ##                "integer"                 "double"                 "double" 
    ##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
    ##                "integer"                 "double"                 "double" 
    ##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
    ##                "integer"                 "double"                 "double" 
    ##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
    ##                "integer"                "integer"                 "double" 
    ##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
    ##                 "double"                 "double"                 "double" 
    ##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
    ##                 "double"                 "double"                 "double" 
    ##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
    ##                 "double"                 "double"                 "double" 
    ##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
    ##                 "double"                 "double"                 "double" 
    ##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
    ##                "integer"                "integer"                "integer" 
    ##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
    ##                "integer"                 "double"                 "double" 
    ##                   classe 
    ##                "integer"

Lets check for the frequency of NAs

``` r
#Training data: compute frequency of NA per column
dataTrainNA <- round(colSums(is.na(dataTrain)|dataTrain=='')/nrow(dataTrain),2)
dataTrainNA 
```

    ##                        X                user_name     raw_timestamp_part_1 
    ##                     0.00                     0.00                     0.00 
    ##     raw_timestamp_part_2           cvtd_timestamp               new_window 
    ##                     0.00                     0.00                     0.00 
    ##               num_window                roll_belt               pitch_belt 
    ##                     0.00                     0.00                     0.00 
    ##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
    ##                     0.00                     0.00                     0.98 
    ##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
    ##                     0.98                     0.98                     0.98 
    ##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
    ##                     0.98                     0.98                     0.98 
    ##           max_picth_belt             max_yaw_belt            min_roll_belt 
    ##                     0.98                     0.98                     0.98 
    ##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
    ##                     0.98                     0.98                     0.98 
    ##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
    ##                     0.98                     0.98                     0.98 
    ##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
    ##                     0.98                     0.98                     0.98 
    ##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
    ##                     0.98                     0.98                     0.98 
    ##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
    ##                     0.98                     0.98                     0.98 
    ##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
    ##                     0.00                     0.00                     0.00 
    ##             accel_belt_x             accel_belt_y             accel_belt_z 
    ##                     0.00                     0.00                     0.00 
    ##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
    ##                     0.00                     0.00                     0.00 
    ##                 roll_arm                pitch_arm                  yaw_arm 
    ##                     0.00                     0.00                     0.00 
    ##          total_accel_arm            var_accel_arm             avg_roll_arm 
    ##                     0.00                     0.98                     0.98 
    ##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
    ##                     0.98                     0.98                     0.98 
    ##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
    ##                     0.98                     0.98                     0.98 
    ##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
    ##                     0.98                     0.98                     0.00 
    ##              gyros_arm_y              gyros_arm_z              accel_arm_x 
    ##                     0.00                     0.00                     0.00 
    ##              accel_arm_y              accel_arm_z             magnet_arm_x 
    ##                     0.00                     0.00                     0.00 
    ##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
    ##                     0.00                     0.00                     0.98 
    ##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
    ##                     0.98                     0.98                     0.98 
    ##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
    ##                     0.98                     0.98                     0.98 
    ##            max_picth_arm              max_yaw_arm             min_roll_arm 
    ##                     0.98                     0.98                     0.98 
    ##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
    ##                     0.98                     0.98                     0.98 
    ##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
    ##                     0.98                     0.98                     0.00 
    ##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
    ##                     0.00                     0.00                     0.98 
    ##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
    ##                     0.98                     0.98                     0.98 
    ##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
    ##                     0.98                     0.98                     0.98 
    ##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
    ##                     0.98                     0.98                     0.98 
    ##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
    ##                     0.98                     0.98                     0.98 
    ## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
    ##                     0.98                     0.98                     0.00 
    ##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
    ##                     0.98                     0.98                     0.98 
    ##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
    ##                     0.98                     0.98                     0.98 
    ##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
    ##                     0.98                     0.98                     0.98 
    ##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
    ##                     0.98                     0.00                     0.00 
    ##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
    ##                     0.00                     0.00                     0.00 
    ##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
    ##                     0.00                     0.00                     0.00 
    ##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
    ##                     0.00                     0.00                     0.00 
    ##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
    ##                     0.00                     0.98                     0.98 
    ##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
    ##                     0.98                     0.98                     0.98 
    ##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
    ##                     0.98                     0.98                     0.98 
    ##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
    ##                     0.98                     0.98                     0.98 
    ##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
    ##                     0.98                     0.98                     0.98 
    ##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
    ##                     0.98                     0.00                     0.98 
    ##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
    ##                     0.98                     0.98                     0.98 
    ##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
    ##                     0.98                     0.98                     0.98 
    ##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
    ##                     0.98                     0.98                     0.98 
    ##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
    ##                     0.00                     0.00                     0.00 
    ##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
    ##                     0.00                     0.00                     0.00 
    ##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
    ##                     0.00                     0.00                     0.00 
    ##                   classe 
    ##                     0.00

The columns with NAs cannot get used at all since they contain almost nothing else; they will be excluded from both the training and testing data

Select the features to keep for the analysis
--------------------------------------------

In addition to these columns with NAs, The first 7 columns that contain non relevant data to the analysis (not measures of a device) get excluded as well

``` r
#exclude all the columns with NA
trainDataset <- dataTrain[,names(dataTrainNA)[dataTrainNA==0]]
#exclude the first columns with user, timestamps and window data
trainDataset <- trainDataset[, -c(1,2,3,4,5,6,7)]

#Testing data: keep only the columns selected in the training set, excluding the last one that is the predicted column "classe"
col <-  as.vector(names(trainDataset))[-53]
testDataset <- dataTest[, col]
```

The original training data is split into training and cross validation sets in order to evaluate the accuracy of the model prediction

``` r
#split the original training data into training and cross validation sets
inTrain = createDataPartition(trainDataset$classe, p = 3/4)[[1]]
training = trainDataset[ inTrain,]
xvalidation = trainDataset[-inTrain,]
```

Build the model using the new training set
------------------------------------------

Since the problem is about classification, the retained model is a Random Forest which is based on bagging to reduce the variance so that we can get good performance on the unseen data. The quality of the predictions is assessed against the cross validation data set (out of sample data) using a confusion matrix:

``` r
modFit <- randomForest(classe ~ ., data = training)
modXValPredict <- predict(modFit, xvalidation )
confusionMatrix(xvalidation$classe, modXValPredict)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    0    0    0    0
    ##          B    6  942    1    0    0
    ##          C    0    2  853    0    0
    ##          D    0    0    4  799    1
    ##          E    0    0    0    5  896
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9961         
    ##                  95% CI : (0.994, 0.9977)
    ##     No Information Rate : 0.2857         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9951         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9957   0.9979   0.9942   0.9938   0.9989
    ## Specificity            1.0000   0.9982   0.9995   0.9988   0.9988
    ## Pos Pred Value         1.0000   0.9926   0.9977   0.9938   0.9945
    ## Neg Pred Value         0.9983   0.9995   0.9988   0.9988   0.9998
    ## Prevalence             0.2857   0.1925   0.1750   0.1639   0.1829
    ## Detection Rate         0.2845   0.1921   0.1739   0.1629   0.1827
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9979   0.9981   0.9968   0.9963   0.9988

The accuracy is good, hence we keep this model for the predictions.

Use the model to make the predictions
-------------------------------------

Finally the model is applied to the original testing data set in order to make the predictions (Classe):

``` r
# make the prediction on the test data
predict(modFit, testDataset )
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Those are the numbers entered into tzhe "Course Project Prediction Quiz"."
