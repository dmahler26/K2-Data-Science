
# NYC Rental Listing Data - MVP Analysis

## Introduction

This project is inspired from the [NYC Rental Listing Kaggle challenge](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries). The original goal of the challenge was to develop a means of predicting interest level in new listings using the rental listing data provided by [RentHop](https://www.renthop.com/).

For the purposes of this project, however, we will simply be exploring the data and tring to try and provide a thorough investigation to the following question:
> **What are the key influencers of the interest level for a listing?**
 
As part of the [K2 Data Science](http://www.k2datascience.com/) curriculum, the focus of this data exploration will be on data cleaning, basic numerical and statistical analysis, and data visualization. This open-ended project should lay the foundation for future analysis in returning to Kaggle's original prompt of building a method for predicting interst level in a given rental listing.

This notebook will begin with exploring the basics of the dataset and performing a high level analysis of what is assumed to be one of the key influencers in interest level: rental price.

## The Dataset

The [training dataset](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data) provided by Kaggle & RentHop will be the focus of this project. The file is in JSON format and will be imported and manipulated using the Pandas library. Each row in the dataset constitutes a rental listing with the following information:

1. **building_id**: ID for the building in which a rental is located. Not a unique ID since there can be many rentals per building.
2. **listing_id**: unique ID for the RentHop listing
2. **manager_id**: ID for the manager of listing/building
3. **created**: date the listing was created.
4. **display_address**: simplified street address, typically reduced to the major street name(s)
5. **street_address**: specific street address with house number etc.
6. **bedrooms**: number of bedrooms
7. **bathrooms**: numberof bathrooms
8. **price**: rent per month in USD
9. **latitude**: latitude of property
10. **longitude**: longitude of property
11. **features**: list of features (e.g. pet friendly, A/C, parking)
12. **description**: written description of the rental
12. **photos**: list of image URLs
13. **interest_level**: interest in the listing. Has 3 categories: **'low', 'medium', 'high'**

A total of **49352 records** are present in the origian dataset.

## Data Cleaning

(See [data cleaning notebook](nyc_re_cleaning.ipynb) for details)

Intitial review of the data highlights a few key areas of interest in which data may need to be fixed or removed:
1. Price
2. Location

For rental prices, the main concerns are that there are listings with incorrect, inflated prices (i.e. buy vs. rent), were priced as an entire building or commercial rental vs. a residence, or were priced unrealistically low given the rental's attributes. The following steps and assumptions were taken to try and identify and remove outliers from the data set:

1. For records listed as more than \$30,000, is there enough detail provided to justify this price? Is the listing for an entire building? Is it a commerical or retail space? (9 records removed)
2. For records listed as 0 bedroom priced over \$3,000, is the listing for a commercial or retail space? (4 records removed)
3. For records listed as less than $1000, is there enough detail provided to justify this price? (2 records removed)

For location, beyond simply identifying the instances where latitude/longitude was missing (i.e. coordinates of [0.0, 0.0]), there were a signficiant number of coordinates that appeared to fall outside the NYC area. The following approach was taken to remove location outliers:
1. Is location missing? (12 records removed)
2. Is the location beyond 2-3 standard deviations from the mean location? Are these results located outside of NYC and its buroughs? (37 records removed)

With these steps complete, the majority of outliers in both price and location should have been identified and discarded, reducing the dataset from 49,352 to 49,288 records

## Initial Data Exploration

(See [MVP analysis notebook](mvp_analysis.ipynb) for details)

### Interest Level

The first step is to investigate the key attribute of interest: interest level.

Summing the number of records in each interest level yields the following results:

![MVP_BarChart_NumberOfListingsPerInterestLevel.png](figures\MVP_BarChart_NumberOfListingsPerInterestLevel.png)

It is clear that low interest level listings dominate the data set, whilst high interest level listings appear to be somewhat of a rare occurance. Calculating the actual numbers for these counts yields the following proportions: 

|Interest Level|Count|Proportion|
|:-:|:-:|:-:|
| Low | 34,228 | 69.4% |
| Medium | 11,225 | 22.8% |
| High | 3835 | 7.8% |

Only 7.8% of listings are high interest. It will be important to keep this proportion in consideration for future calculations, since it may be necessary to weigh results according to their likelihood in order to amplify the signficance of a listig having high interest, and get visually signficiant results when analyzing relations to high interest.

### Price

The most intuitive influencer one would expect to have on a given listings interest level is price. Not only is budget is likely to be one of the most limiting and uncompromisable factors in a search for a home, but one would expect the ability to save money with cheaper rentals would also attract more interest.

The following distribution plots were performed to some initial insight into the data set:

![MVP_KDE_Price.png](figures\MVP_KDE_Price.png)

Due to some of the high priced outliers, the first plot shows an extremely skewed density plot, with a mean price of \$3,664 and the majority of record falling between the \$1,500-\$10,000 range. A more useful view is acheived when focusing on those listings below \$10,000. However, even with this reduced set we obtain a skew of 1.45, so the application of any normal statistics would yield biased results and therefore should be limited.

A better analysis is likely to be acheived focusing on the median and quartile ranges, which should lend focus to this core range of prices. A boxplot of price across each interest levels is one such method (note that this boxplot ignores outliers for a more useful view):

![MVP_Boxplot_PriceByInterestLevelNoFliers.png]("figures/MVP_Boxplot_PriceByInterestLevelNoFliers.png")

There is a visible decrease in price as we progress from low to high interest, which is the first confirmation of the initial hypothesis than lower prices would be more attractive. Comparing the medians for each group to the population median yields the following: <br>
<br>

| Interest Level | Median Price | Difference from Pop. Median | % Difference |
|:-:|:-:|:-:|:-:|
|Low|33,300|+150|+5%|
|Medium|2,895|-255|-8%|
|High|2,400|-750|-24%|

The median for high interest level listings is 24% less than that of the population median, which gives a bit more weight to observed trend in the boxplot. However, there is still much overlap in the range of prices observed in each interest level. This suggests that, whilst price may be one of the influencers in interest level, there is a definite need to explore additonal factors as price alone is not enough.

In addition to the boxplots above, it is useful to see the proportions of where listings fall in terms of price for each interest level. The first approach taken was determining which standard deviation each price fell within, however as mentioned earlier due to the skeweness in price this did not yield much useful information. Instead, splitting price into 4 quartiles (25th, 50th, 75th, 100th) gave a much clearer picture:

![MVP_Heatmap_PriceQuartilePerInterestLevel.png]("figures/MVP_Heatmap_PriceQuartilePerInterestLevel.png")

From the heatmap above one can see that a clear majority of high interest listings fall within 1st (lowest) price quartile, whereas majority of low interest in the 3rd and 4th quartile. This helps confirm that having lower price is a key attribute for high interest, even if the differences in the boxplots are not as significant as one might have expected.

## Next Steps

Following this initial analysis of price and interest level, there is a lot left to explore within the data for possible influencers on interest level.

Firstly, a more in-depth analysis of price & interest level should be conducted which takes into account other factors/ratios such as:
* Price per bedrooms
* Price per bathroom
* Price per total number of rooms

These ratios should hopefully better represent the actual value in a given listing, which in turn should have a greater influence interest level than price alone.

Secondly, there is the other key component in real estate which has yet to be explored: location. The following are some of the possible avenues in which the relationship between location and interest level could explored:
* Distribution of low, medium, high interest throughout the city. Areas of concentration.
* Analysis of interest level by Burough
* Analysis of interest level vs. surrounding listings (e.g. price, interest level, features of nearby listings)

Finally, the set of 'features' and other non-numerical attributes are left, with the following possible correlations:
* Frequency of various features across interest levels
* Presence of photos, number of photos
* Presence of description, length of description
* Presence of features, number of features
