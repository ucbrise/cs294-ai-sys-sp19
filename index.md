---
layout: default
---


# AI-Sys Spring 2019

* **When**: *Mondays and Wednesdays from 9:30 to 11:00*
* **Where**: *Soda 405*
* **Instructors**: [Ion Stoica]() and [Joseph E. Gonzalez](https://eecs.berkeley.edu/~jegonzal)
* **Announcements**: [Piazza](https://piazza.com/berkeley/spring2019/cs294159/home)
* **Sign-up to Present**: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1NLLVPh8QioXRtzYEKc3XjtJMLqbT8WMMQ27bQz8lSJI/edit?usp=sharing
)
* **Project Ideas**: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/16Oz8ZJ0x1AdukWQxq7QYdzkzoVH70vbhtSOGlJ_EFKc/edit#gid=0)
* If you have reading suggestions please send a pull request to this course website on [Github](https://github.com/ucbrise/cs294-ai-sys-sp19) by modifying the [index.md](https://github.com/ucbrise/cs294-ai-sys-sp19/blob/master/index.md) file.


## Course Description
<blockquote class="blockquote">
<p>
The recent success of AI has been in large part due in part to advances in hardware and software systems. 
These systems have enabled training increasingly complex models on ever larger datasets. In the process, these systems have also simplified model development, enabling the rapid growth in the machine learning community. 
These new hardware and software systems include a new generation of GPUs and hardware accelerators (e.g., TPU and Nervana), open source frameworks such as Theano, TensorFlow, PyTorch, MXNet, Apache Spark, Clipper, Horovod, and Ray, and a myriad of systems deployed internally at companies just to name a few. 
At the same time, we are witnessing a flurry of ML/RL applications to improve hardware and system designs, job scheduling, program synthesis, and circuit layouts.  
</p>

<p>  
In this course, we will describe the latest trends in systems designs to better support the next generation of AI applications, and applications of AI to optimize the architecture and the performance of systems. 
The format of this course will be a mix of lectures, seminar-style discussions, and student presentations. 
Students will be responsible for paper readings, and completing a hands-on project. Readings will be selected from recent conference proceedings and journals. 
For projects, we will strongly encourage teams that contains both AI and systems students.
</p>
</blockquote>



## Course Syllabus




<!-- This is the dates for all the lectures -->
{% capture dates %}
1/23/19
1/28/19
1/30/19
2/4/19
2/6/19
2/11/19
2/13/19
2/18/19
2/20/19
2/25/19
2/27/19
3/4/19
3/6/19
3/11/19
3/13/19
3/18/19
3/20/19
3/25/19
3/27/19
4/1/19
4/3/19
4/8/19
4/10/19
4/15/19
4/17/19
4/22/19
4/24/19
4/29/19
5/1/19
5/6/19
5/8/19
{% endcapture %}
{% assign dates = dates | split: " " %}

This is a tentative schedule.  Specific readings are subject to change as new material is published.

<a href="#today"> Jump to Today </a>

<table class="table table-striped syllabus">
<thead>
   <tr>
      <th style="width: 5%"> Week </th>
      <th style="width: 10%"> Date (Lec.) </th>
      <th style="width: 85%"> Topic </th>
   </tr>
</thead>
<tbody>








{% include syllabus_entry %}
## Introduction and Course Overview

This lecture will be an overview of the class, requirements, and an introduction to what makes great AI-Systems research.

### Slide Links
* Course Overview [[pdf](assets/lectures/l1.pdf), [pptx](assets/lectures/l1.pptx)]







{% include syllabus_entry %}
## Convolutional Neural Network Architectures


**Minor Update:** We have moved the reading on auto-encoders to Wednesday.

Reading notes for the two required readings below must be submitted using this **[google form](https://goo.gl/forms/BDHKbtmypsw9UPyj2)** by Monday the 28th at 9:30AM. We have asked that for each reading you answer the following questions:
1. What is the problem that is being solved?
1. What are the metrics of success?
1. What are the key innovations over prior work?
1. What are the key results?
1. What are some of the limitations and how might this work be improved?
1. How might this work have long term impact?

If you find some of the reading confusing and want a more gentle introduction, the optional reading contains some useful explanatory blog posts that may help.

### Links
* [Reading Quiz](https://goo.gl/forms/BDHKbtmypsw9UPyj2) due before class.
* Intro Lecture + AlexNet [[pdf](assets/lectures/lec02/convolutional_networks_v2.pdf), [pptx](assets/lectures/lec02/convolutional_networks_v2.pptx)]
* Classic Neural Architectures and Inception-v4 [[pdf](assets/lectures/lec02/classic_neural_architectures.pdf), [pptx](assets/lectures/lec02/classic_neural_architectures.pptx)]



<div class="reading">
<div class="required_reading" markdown="1">


* The [AlexNet paper](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) that both help launch deep learning and also advocate for systems and ML.  Take a look at how system constraints affected the model.
* [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261).  In retrospect, the paper [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) provides a better overview of the ideas and motivations behind the latest inception models.


</div>
<div class="optional_reading" markdown="1">

### Convolutional Networks

* For a quick introduction to convolutional networks take a look at [CS231 Intro to Convolutional Networks](http://cs231n.github.io/convolutional-networks/) and [Chris Olah's illustrated posts](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/).
* Much of contemporary computer vision can be traced back to the original [LeNet paper](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) and it's corresponding [90's era website](http://yann.lecun.com/exdb/lenet/).
* There is a line of work that builds on [residual networks](https://arxiv.org/abs/1512.03385) starting with [Highway Networks](https://arxiv.org/abs/1505.00387), then [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), and then more recently [Deep Layer Aggregation](https://arxiv.org/abs/1707.06484).  This [blog post](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035) provides a nice overview.


</div>
</div>
















{% include syllabus_entry %}
## More Neural Network Architectures


### Links
* [Reading Quiz](https://goo.gl/forms/POkEId1xA5c9FjoI3) due before class. 
* Intro [[pdf](assets/lectures/lec03/other_networks.pdf), [pptx](assets/lectures/lec03/other_networks.pptx)]
* Autoencoders [[pdf](assets/lectures/lec03/autoencoders.pdf), [pptx](assets/lectures/lec03/autoencoders.pptx)]


<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">


* We had originally assigned, [Autoencoders, Unsupervised Learning, and Deep Architectures](http://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf).  However this paper is a bit theoretical for the goals of this class. Instead, you may alternatively read [this overview paper](https://arxiv.org/pdf/1801.01586.pdf) and use it when filling in the reading form.
* [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)



</div>
<div class="optional_reading" markdown="1">





### Auto-Encoders

* An excellent [Survey on Autoencoders](https://www.doc.ic.ac.uk/~js4416/163/website/)
* A tutorial on [variational auto-encoders](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) (and another [tutorial](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf))
* Original work on auto-encoders [Learning Internal Representations by Error Propagation](https://ieeexplore.ieee.org/document/6302929) by Rumelhart and McClelland.


### Graph Networks

* The paper ["Relational inductive biases, deep learning, and graph networks"](https://arxiv.org/abs/1806.01261) provides some background and motivations behind deep learning on relational objects and introduces a general **Graph Network** framework.
* The paper ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907) introduces graph convolutional networks.

<!-- ### Recurrent Neural Networks

### Language Networks:
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)



* Andrej Karpathy has an excellent overview [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on RNNs.
* Chris Olah has a [well illustrated overview of LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).


### Capsule Networks
* A nice [overview](https://medium.com/ai³-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) of Capsule networks.

### Other Kinds of Data
* [Geometric deep learning: going beyond Euclidean data](https://arxiv.org/abs/1611.08097)
 -->

</div>
</div>







{% include syllabus_entry %}
## Deep Learning Frameworks 

### Links
* [Reading Quiz](https://goo.gl/forms/vxGNPZ9HK99Yl6QI2) due before class. 
* Intro Lecture [[pdf](assets/lectures/lec04/lec04.pdf), [pptx](assets/lectures/lec04/lec04.pptx)]
* TensorFlow Presentation [[pdf](assets/lectures/lec04/tf.pdf), [pptx](assets/lectures/lec04/tf.pptx)]



<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](http://download.tensorflow.org/paper/whitepaper2015.pdf) and/or [TensorFlow OSDI Paper](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)
* [MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://www.cs.cmu.edu/~muli/file/mxnet-learning-sys.pdf)

</div> 
<div class="optional_reading" markdown="1">
* The following [Comparative Study of Deep Learning Software Frameworks](https://arxiv.org/pdf/1511.06435.pdf) provides a good (but a little dated) comparison of the various frameworks.
* [Automatic differentiation in PyTorch](https://openreview.net/pdf?id=BJJsrmfCZ)
* A more detailed overview of [Theano](https://arxiv.org/pdf/1605.02688.pdf).
</div>
</div>










{% include syllabus_entry %}
## RL Systems & Algorithms


### Links
* [Reading Quiz](https://goo.gl/forms/awQKZjxtb1PV0g272) due before class. 
* RLlib [[pdf](assets/lectures/rllib.pdf)]
* A3C [[pdf](assets/lectures/A3C.pdf)]

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">
* [Asynchronous Methods for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/mniha16.pdf)
* [RLlib: Abstractions for Distributed Reinforcement Learning](https://arxiv.org/abs/1712.09381)

</div> 
<div class="optional_reading" markdown="1">

* [Horizon: Facebook's Open Source Applied Reinforcement Learning Platform](https://arxiv.org/abs/1811.00260)

</div>
</div>







{% include syllabus_entry %}
## Application: Data Structure and Algorithms

### Links
* [Reading Quiz](https://goo.gl/forms/uKUtXqhpv2Jctqgq1
) due before class. 
* Learned Indexes [[pdf](assets/lectures/lec05/learnedIndexes.pdf), [pptx](assets/lectures/lec05/learnedIndexes.pptx)]
* Learning to Optimize Join Queries [[pdf](assets/lectures/lec05/dq.pdf)]
<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208)
* [Learning to Optimize Join Queries With Deep Reinforcement Learning](https://arxiv.org/abs/1808.03196)

</div>

<div class="optional_reading" markdown="1">

* [SageDB: A Learned Database System](http://alexbeutel.com/papers/CIDR2019_SageDB.pdf)
* [RLgraph: Flexible Computation Graphs for Deep Reinforcement Learning](https://arxiv.org/1810.09028)

</div>
</div>








{% include syllabus_entry %}
## Distributed Systems for ML

### Links
* [Reading Quiz](https://goo.gl/forms/aalfGnuX1ZFdYhlX2) due before class. 
* Learned Cardinalities [[pdf](assets/lectures/lec05/learned-cardinalities.pdf)]



<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208) cont'd
* [Learned Cardinalities: Estimating Correlated Joins with Deep Learning](https://arxiv.org/pdf/1809.00677.pdf)

</div>
</div>







{% include syllabus_entry %}

<center><h1>Administrative Holiday (Feb 18th)</h1></center> 







{% include syllabus_entry %}
## Hyperparameter search



### Links
* [Reading Quiz](https://goo.gl/forms/z7CcNYyXD9rL67pI2) due before class.  There was a mix-up in updating the reading and the wrong paper was swapped.  You may either read the Hyperband paper (preferred) or the Vizer paper (see optional reading) for the second reading. 
* A Generalized Framework for Population Based Training [[pdf](assets/lectures/lec09/generalized-PBT.pdf)]


<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [A Generalized Framework for Population Based Training](https://arxiv.org/pdf/1902.01894.pdf)
* [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/pdf/1603.06560.pdf)

</div>


<div class="optional_reading" markdown="1">

 
* [Google Vizier: A Service for Black-Box Optimization](https://research.google.com/pubs/archive/46180.pdf)

</div>


</div>







{% include syllabus_entry %}

## Auto ML & Neural Architecture Search (1/2)

### Links
* [Reading Quiz](https://goo.gl/forms/Twa0EQ9rJKIZ0eaj1) due before class.
* AutoML Overview [[pdf](assets/lectures/lec10/automl.pdf), [pptx](assets/lectures/lec10/automl.pptx)]

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Efficient and Robust Automated Machine Learning](https://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-preprint.pdf)
* [Designing Neural Network Architectures using Reinforcement Learning](https://arxiv.org/abs/1611.02167)
`
</div>
</div>







{% include syllabus_entry %}
## Auto ML & Neural Architecture Search (2/2)

### Links
* [Reading Quiz](https://goo.gl/forms/xOVySnneBUDiNjAC3) due before class.

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268)
* [Searching for Efficient Multi-Scale Architectures for Dense Image Prediction](https://arxiv.org/abs/1809.04184)

</div>
</div>





{% include syllabus_entry %}
## Autonomous Vehicles

### Links
* [Reading Quiz](https://goo.gl/forms/Ai4L7UFj3YD40YJJ2) due before class.

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [The Architectural Implications of Autonomous Driving](https://web.eecs.umich.edu/~shihclin/papers/AutonomousCar-ASPLOS18.pdf)
* [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079)

</div>
<div class="optional_reading" markdown="1">

* [Software Infrastructure for an Autonomous Ground Vehicle](https://www.ri.cmu.edu/pub_files/2008/12/TartanInfrastructure.pdf)
* [Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)

</div>
</div>





{% include syllabus_entry %}
## Deep Learning Compilers

### Links
* [Reading Quiz](https://goo.gl/forms/Bf6Qobcj4QkizOXu2) due before class.

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)
* [TensorComprehensions](https://arxiv.org/abs/1802.04730)

</div>
<div class="optional_reading" markdown="1">

* [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166): The TVM story is two fold. There's a System for ML story (above paper) and this paper is their the ML for System story.

</div>
</div>






{% include syllabus_entry %}
<center> <h1>Project Presentation Checkpoints</h1> </center>





{% include syllabus_entry %}
## Application: Program synthesis
<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Learning to Represent Programs with Graphs](https://openreview.net/forum?id=BJOFETxR-)
* [DeepCoder: Learning to write programs](https://openreview.net/pdf?id=ByldLrqlx)

</div>
</div>



{% include syllabus_entry %}
## Distributed Deep Learning  (Part 1)
<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
* [Large Scale Distributed Deep Networks](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)

</div>

<div class="optional_reading" markdown="1">

* [Exascale Deep Learning for Climate Analytics](https://arxiv.org/abs/1810.01993)
* [ImageNet/ResNet-50 Training in 224 Seconds](https://arxiv.org/abs/1811.05233)

</div>
</div>





{% include syllabus_entry %}
## Distributed Deep Learning (Part 2)
<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)
* [Scaling Distributed Machine Learning with the Parameter Server](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf)

</div>
</div>


{% include syllabus_entry %}
<center><h1>Spring Break (March 25th)</h1></center> 

{% include syllabus_entry %}
<center><h1>Spring Break (March 27th)</h1></center> 








{% include syllabus_entry %}
## Application: Networking

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Neural Adaptive Video Streaming with Pensieve - People - MIT](https://people.csail.mit.edu/hongzi/content/publications/Pensieve-Sigcomm17.pdf)
* [Internet Congestion Control via Deep Reinforcement Learning]()

</div>
<div class="optional_reading" markdown="1">

* [PCC Vivace: Online-Learning Congestion Control](https://arxiv.org/abs/1811.00260)

</div>
</div>







{% include syllabus_entry %}
## Dynamic Neural Networks

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Outrageously large neural networks: The sparsely-gated mixture-of-experts layer](https://arxiv.org/abs/1701.06538)
* [SkipNet: Learning Dynamic Routing in Convolutional Networks](https://arxiv.org/pdf/1711.09485.pdf)

</div>
</div>










{% include syllabus_entry %}
## Model compression for edge devices

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and less than 0.5MB model size](https://arxiv.org/abs/1602.07360)
* [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

</div>
<div class="optional_reading" markdown="1">

* [MobileNetV2: Inverted Residuals and Linear Bottlenecks]()

</div>
</div>









{% include syllabus_entry %}
## Applications: Security

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Helen](https://arxiv.org/abs/1602.07360)
* [Federated Learning: Strategies for Improving Communication Efficiency](https://ai.google/research/pubs/pub45648)

</div>
<div class="optional_reading" markdown="1">

* [SecureML: A System for Scalable Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/396.pdf)

</div>
</div>












{% include syllabus_entry %}
## Application: Prediction Serving

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications](https://arxiv.org/abs/1811.09886)
* [Clipper: A Low-Latency Online Prediction Serving System](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf)


</div>
<div class="optional_reading" markdown="1">

* [TFX: A TensorFlow-Based Production-Scale Machine Learning Platform](https://www.kdd.org/kdd2017/papers/view/tfx-a-tensorflow-based-production-scale-machine-learning-platform)
* [TensorFlow-Serving: Flexible, High-Performance ML Serving](https://arxiv.org/abs/1712.06139)


</div>
</div>










{% include syllabus_entry %}
## Distributed RL Algorithms

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
* [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)

</div>
</div>













{% include syllabus_entry %}
## Explanability & Interpretability

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
* [The Mythos of Model Interpretability](https://arxiv.org/pdf/1606.03490.pdf)

</div>
<div class="optional_reading" markdown="1">

* [Grad-CAM](https://arxiv.org/abs/1610.02391)

</div>
</div>








{% include syllabus_entry %}
## Scheduling for DL Workloads

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Optimus: an efficient dynamic resource scheduler for deep learning clusters](https://blog.acolyer.org/2018/06/12/optimus-an-efficient-dynamic-resource-scheduler-for-deep-learning-clusters/)
* [Gandiva: Introspective Cluster Scheduling for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/xiao)

</div>
<div class="optional_reading" markdown="1">

* [Grad-CAM](https://arxiv.org/abs/1610.02391)

</div>
</div>







{% include syllabus_entry %}
## New Neural Architectures

<div class="summary" markdown="1"> </div>
<div class="reading">
<div class="required_reading" markdown="1">

* [Matrix capsules with EM Routing](https://ai.google/research/pubs/pub46653)
* [Cortical Learning via Prediction](http://proceedings.mlr.press/v40/Papadimitriou15.pdf)

</div>
</div>





{% include syllabus_entry %}
<center> <h1>Class Summary</h1> </center>





{% include syllabus_entry %}
<center> <h1>RRR Week (May 7th)</h1> </center>
{% include syllabus_entry %}
<center> <h1>RRR Week (May 9th)</h1> </center>











</td>
</tr>
</tbody>
</table>






## Projects

Detailed candidate project descriptions will be posted shortly.  However, students are encourage to find projects that relate to their ongoing research.


## Grading

Grades will be largely based on class participation and projects.  In addition, we will require weekly paper summaries submitted before class.
* **Projects:** _60%_
* **Weekly Summaries:** _20%_
* **Class Participation:** _20%_









<script type="text/javascript">


var current_date = new Date();
var rows = document.getElementsByTagName("th");
var finished =  false;
for (var i = 1; i < rows.length && !finished; i++) {
   var r = rows[i];
   if (r.id.startsWith("counter_")) {
      var fields = r.id.split("_")
      var week_div_id = "week_" + fields[2]
      var lecture_date = new Date(fields[1] + " 23:59:00")
      if (current_date <= lecture_date) {
         finished = true;
         r.style.background = "orange"
         r.style.color = "black"
         var week_td = document.getElementById(week_div_id)
         week_td.style.background = "#043361"
         week_td.style.color = "white"
         var anchor = document.createElement("div")
         anchor.setAttribute("id", "today")
         week_td.prepend(anchor)
      }
   }
}

$(".reading").each(function(ind, elem) {
   var optional_reading = $(elem).find(".optional_reading");
   if(optional_reading.length == 1) {
      optional_reading = optional_reading[0];
      optional_reading.setAttribute("id", "optional_reading_" + ind);
      var button = document.createElement("button");
      button.setAttribute("class", "btn btn-primary btn-sm");
      button.setAttribute("type", "button");
      button.setAttribute("data-toggle", "collapse");
      button.setAttribute("data-target", "#optional_reading_" + ind);
      button.setAttribute("aria-expanded", "false");
      button.setAttribute("aria-controls", "#optional_reading_" + ind);
      optional_reading.setAttribute("class", "optional_reading_no_heading collapse")
      button.innerHTML = "Additional Optional Reading";
      optional_reading.before(button)
   }
   
})


</script>


