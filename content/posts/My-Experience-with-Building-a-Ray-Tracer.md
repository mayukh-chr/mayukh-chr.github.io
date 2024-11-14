+++
title = "My Experience with Building a Ray Tracer"
date = "2024-11-01"
draft = false
+++

# Introduction

&emsp; &emsp; Despite how my [GitHub contributions](http://github.com/mayukh-chr) might look, my preferred programming language is actually C++. I’ve used it extensively for Leetcode and Codeforces since learning it about a year ago. My reasons for choosing C++ are its convenient STL library, faster runtime (compared to Python), and relatively concise syntax (unlike Java and C), making it a top choice for competitive programmers. Naturally, after working with the fundamentals for so long, I decided to take on a project in C++.

# The Project

&emsp; &emsp; I chose to work through the book [*Ray Tracing in One Weekend*](https://raytracing.github.io/books/RayTracingInOneWeekend.html) by [Peter Shirley](https://github.com/petershirley), [Trevor David Black](https://github.com/trevordblack), and [Steve Hollasch](https://github.com/trevordblack). This book is the first in a trilogy, all available for free, and it includes code snippets that help when you get stuck. 

&emsp; &emsp; I wanted to work on something related to game engines, a domain I hadn’t yet explored. Since Nvidia’s 2018 RTX announcement, I’ve been intrigued by ray tracing and wanted to experiment with it. However, without access to the necessary GPUs, I had to leave RTX-specific experimentation aside for now.

# What I Learned

&emsp; &emsp; The book starts with generating a simple RGB image in PPM format and gradually introduces features like scenes, cameras, and objects. In the second half, it simulates light rays hitting the camera with effects like reflections, refractions, diffusion, anti-aliasing, and even lens blur.

$$ From \space this $$
![img1](/images/p3/img-1.03-red-sphere.png)
$$ to \space this $$
![img2](/images/p3/img-1.23-book1-final.jpg)

## Challenges I Faced

&emsp; &emsp; My competitive programming experience in C++ was limited to using the STL, with little exposure to Object-Oriented Programming (OOP) concepts. This project was my first experience with OOP in C++, and I found it challenging. No matter how much OOP code I write, I can’t seem to fully get comfortable with it. Interestingly, I found my coursework in Java (which is more explicit in its syntax) to be a bit easier. I had to look up several terms I hadn’t encountered before, as some weren’t covered in my OOP classes.

&emsp; &emsp; Additionally, I was working with many custom data types (e.g., vectors, points, surfaces, materials) for the first time, which took time to adjust to, as I was mostly familiar with predefined types.

# Future Directions

&emsp; &emsp; I skimmed through the next two parts of the book and, although I’m very interested in continuing the project, I feel I'll end up with the same feelings after finishing it. I attempted to implement multithreading to speed up rendering, but I struggled with thread synchronization during rendering. I’d love to try implementing the project with [CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) someday when I have access to the necessary hardware.

&emsp; &emsp; I do want to work in low level systems wiht C++ (or even rust), because in my opinion that's a domain it has a slow developmental cycle a lot of memory sensitive procedures, making sure the end product is as tight as possible in terms of response times and memory handling. 
