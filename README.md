# Gene-compression-and-Cancer-type-classification
Fall 2021 Final Project for CM226 - Machine Learning in Bioinformatics \n
Check out the detailed report [here](Group16_ProjectReport.pdf)
## Dataset
* [TCGA](https://portal.gdc.cancer.gov/) -  The Cancer Genome Atlas PanCanAtlas RNAseq data from the National Cancer Institute Genomic Data Commons
* These data consisted of 11,069 samples with 20,531 measured genes.
Preprocessing- 
* Tumors that were measured from multiple sites were removed.
* Data was normalised
* This resulted in a final TCGA PanCanAtlas gene expression matrix with 11,060 samples, which included 33 different cancer types, and 16,148 genes.
* The data is split into 90% training and 10% testing partitions. The data is partitioned such that each split contained relatively equal representation of each cancer type.


## Proposed Model
![alt text](https://github.com/RushiBhatt007/Gene-compression-and-Cancer-type-classification/blob/main/proposed_model.png?raw=true)

## Implementation of PCA, NMF, ICA
* These models have been built using sklearn library

## Implementation of VAE Model
![alt text](https://github.com/RushiBhatt007/Gene-compression-and-Cancer-type-classification/blob/main/vae_model.png?raw=true)

* This VAE model is inspired from [Tybalt's](https://github.com/greenelab/tybalt) implementation

 ## Authors
 * Rushi Bhatt
 * Ronak Kaoshik
 * Shruti Mohanty
 
See also the list of [contributors](https://github.com/RushiBhatt007/Gene-compression-and-Cancer-type-classification/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
