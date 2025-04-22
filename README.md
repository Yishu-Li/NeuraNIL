# NeuraNIL

CSCI1470 Final Project

## Team members

Carlos Ramos: carlos_ramos@brown.edu  
Jania Vandevoorde: jania_vandevoorde@brown.edu  
Yishu Li: yishu_li@brown.edu

CS Logins: jvandevo, cramos6, yli626

## Introduction

Intracortical brain-computer interfaces (iBCIs) hold significant promise for restoring communication and motor functions in patients with conditions such as ALS, stroke, spinal cord injury, etc. However, the instability of neural signals across days remains one of the greatest challenges to achieving long-term stable neural decoding. This project proposes the use of the meta-learning approach, ANIL (Almost No Inner Loop), to enable rapid adaptation to the daily variations in neural signals, enhancing the reliability of neural decoding systems across longer time periods. Furthermore, considering the few-shot learning capabilities of meta-learning, this approach is well-suited for adapting to novel motor imagery types that are not present during the initial training phase.

In this project, we adapt a method from an existing paper, [Raghu et al. (2019)](https://arxiv.org/pdf/1909.09157), for application on neural data. This is a structured prediction problem.

## Related Works

1. **Model agnostic meta-learning:**  
In 2017, [Finn et al.](https://arxiv.org/pdf/1703.03400) proposed a model agnostic meta-learning method (**MAML**) to train a model that can solve different tasks than those it trained on. They achieved this and made the models easy to generalize with only a small number of gradient steps and training data needed to solve the new tasks. This method is compatible with any model trained with gradient descent.  
The training process is divided to two types of parameter updates: *the outer loop* and *inner loop*. The outer loop updates the meta-initialization of the neural network parameters to a setting that enables fast adaptation to new tasks. The inner loop takes the outer loop initialization and performs task-specific adaptation over a few labeled samples.
In 2019, [Raghu et al.](https://arxiv.org/pdf/1909.09157) conjectured that we can obtain the same rapid learning performance of MAML solely through feature reuse by only update the last layer of the model during the inner loop. To test this hypothesis, they introduced **ANIL** (almost no inner loop), a simplified algorithm of MAML that is equally effective but computationally faster. The figures below clarify the difference between the MAML and ANIL methods. 
![Rapid learning and feature reuse](images\image-1.png)
![MAML and ANIL algorithms](images\image.png)

2. **Deep learning, non-linearity, and neural decoding:**  
Linear decoders such as the [Kalman filter](https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf) have been widely used for neural decoding to enable individuals with paralysis to control computer mouses. However, more work is needed to improve this. Previous research shows that deep learning methods such as multi-layer perceptrons and recurrent neural networks can provide higher performance for the task by inducing non-linearity into the decoders. The problem with deep learning is that training models requires a lot more data than simple non-linear decoders. Deep learning models are harder to tune and adapt to new data.

3. **Neural non-stationary and manifold-based iBCI decoders:**  
Previous research has shown the non-stationary of neural signals---models trained on signals from previous days may suffer from performance degradation or even break after several days or weeks. Many methods have been proposed to address this problem, such as manifold-based decoders. These begin by mapping the neural data to a lower dimensional latent space. Then, the latent space neural data is aligned to the initial distribution where the decoder was first trained, eliminating the non-stationary and maintaining the decoder performance across days. This method is shown in the figure below.  
![An illustration of manifold-based iBCI decoder](images\image-2.png)

## Data

1. **FALCON dataset:** The 5 [FALCON](https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1.full.pdf) datasets include neural data recorded form humans, non-human primates and birds. The H2 dataset is a classification task where a human participant imagining writing. The decoder will need to decode the character the participant is trying to write. The dataset is divided three parts: Held-in, Held-out, and and Minival. The Held-in and Minival set are from the training period where the model should be trained and validated and the Held-out set is from the first several blocks in each day from testing period where the model can be fast recelebrated for that day. The figures below are from the FALCON paper and help to visually understand the structure of the datasets. 
![FALCON](images\image-3.png)
![FALCON datasets](images\image-4.png) 

2. **BrainGate2 clinical trial data:** The BrainGate2 clinical trial data we are going to use is from a ongoing clinical trial where a BrainGate2 participant with advanced ALS trying to communicate by imagining a set of gestures. The raw signal was recorded as 30000 Hz voltage data, then preprocessing including re-reference, filtering, threshold crossing, and power calculation will be calculated to get binned spiking rate and spiking power features.

As a meta-learning project, the data will be divided into episodes. Each episode contains a support set and query set. The support set data will be used to update the inner loop while the query data will be used to calculate the loss and update the model.

## Methodology

### Model Structure

The model will be consist of two parts: outer loop (learner) and inner loop (classifier), following the structure of the MAML and ANIL papers. The learner will be fixed after trained on the training data, and the classifier will be updated on each day with *K* step gradient descent.

- **Learner:** To best address the temporal neural data, the learner will be a LSTM model or Transformer model, which takes in the raw neural features and maps the neural data into a lower dimensional latent space to avoid non-stationary.
- **Classifier:** The classifier takes the latent space neural data and makes a prediction. It can be a simple multi-layer perceptron or another machine learning approach such as $k$-nearest neighbor, Gaussian Naive Bayes, or linear discriminant analysis.  

### Training

As mentioned before, each episode of the data will be divided into a support set and a query set. When training, the learner will first perform the forward pass for support set to get the support set latent space data points, and use the support set latent space data points and support set labels to fit the classifier. Then, the forward pass will be performed on query set through both the learner and the fitted classifier to calculate a loss, which will be used to update the parameters in the learner. After the training, we will fix the learner, and refit the classifier on each new days or new tasks. Pseudocode for this approach is shown below. 
![NeuraNIL pseudo code](images\image-5.png)

## Metrics

This is a classification project, so the primary metric will be classification accuracy.

- **Base Goal:** For the BrainGate2 data, there is a known Gaussian Naive Bayes decoded accuracy where the GNB model is trained from scratch everyday. The base goal for this dataset will be to have a similar accuracy compared with the GNB decoder. For the FALCON dataset, the base goal is to approach the example model performance from the FALCON paper.

- **Target Goal:** For both datasets, the target goal is to reach higher accuracy than the given baseline mentioned above. 

- **Stretch Goal:** If we have time, we will try to determine if a smaller recalibration set can be used for the BrainGate2 data.

## Ethics 

- **Why is Deep Learning a good approach to this problem?**  
As mentioned in the previous section, compared with the linear methods being widely used in the iBCI area, deep learning introduces non-linearity to the model, which is more similar to how the brain encodes neural signals. Therefore, as with other tasks, deep learning can give better performance in neural decoding tasks based on its ability to learn complex patterns inn data. However, there are still concerns about the lack of interpretability of deep learning and the question who should be to blame when the algorithm makes mistakes. Some of the mistakes could be serious for stakeholders like ALS patients who may have rely on the iBCI to communicate due to their medical condition. We believe that, in this field, the most urgent concern is still to create a reliable system for  patients in need, so we should prioritize the high performance of deep learning and then try to address sequential problems.

- **Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?**  
The major "stakeholders" in this problem are the people who lost their ability to move or even communicate such as patient with paralysis or ALS. For a patient with late-stage ALS, the iBCI decoder could be their only way to communicate either in their daily life or even for medical communication. Therefore, a unreliable decoder could cause really serious problems. Furthermore, for restoring motor function, if a patient with paralysis is trying to use wheel chair through a iBCI system alone, a decoding mistake may also cause dangerous problems such as making the person fall. 

## Division of Labor (Subject to Change)

- Carlos: lead for FALCON data, model implementation and training
- Jania: lead for written report and poster, model implementation and training
- Yishu: lead for BrainGate2 data, model implementation and training


## Running Environment

Python=3.12  
PyTorch=2.6.0+cu126  
simple-parsing=0.1.7  
wandb=0.19.9  
tqdm=4.67.1  
matplotlib=3.10.1  