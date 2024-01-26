+++
title = "Quake III's Fast Square Root Algorithm"
date = "2024-01-26"
draft = false
+++

# Introduction

&emsp; &emsp; The **Fast inverse square root** or **0x5F3759DF** is an algorithm that approximates $f(x) = 1/\sqrt x$  where $x$ is a 32-bit floating-point number. First observed in the game engine for Quake III Arena in 1999. Which was heavily based on 3D graphics.\

In this piece of writing I will try my best on explaining the mathematical, physical and computational reason of it's existence and why it is such an ingenius hack.\

The reason I am doing this because this algorithm implements concepts from statistical approximation in mathematics, 3D-vectors in theoretical physics, the numerical data storing system in computer architecture, C language, and some bitwise black magic. All of which were all taught to me in my 1st and 2nd years of university, just seperately.\

# The Code

For you impatient, goldfish-brained, tiktok people, here's the function of the code from Quake III Arena, stripped of the C directives (basic boilerplate), with the original comments in place.

```c
float q_rsqrt(float number)
{
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y  = number;
  i  = * ( long * ) &y;                       // evil floating point bit level hacking
  i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
  y  = * ( float * ) &i;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  // y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

  return y;
}
```

# Why does a game engine need this function?

If you want to implement lightng or reflections in your game engine, it is helpful if the vectors in your calculations are *normalized* to have a magnitude 1, if you don't normalise it, things can go wrong during computations.\
\
For normalizing a 3D vector:\
$$\hat{a} = \frac{\vec{A}}{\sqrt{x^2+y^2+z^2}}$$ or \
$$\hat{a} = \vec{A}*\frac{1}{\sqrt{x^2+y^2+z^2}}$$ ie.\

$$\hat{a} = \vec{A}*\frac{1}{|A|}$$

multiplication and addition is easy for computers. For reasons I will mention, the square root is relatively, an extremely slow piece of computation, and division is not much better either.\

For a game engine, this normalisation is carried out over several thousands of surfaces each second. And these slow pieces of computations break the game because it can't catch up with the real-time need of rendering these surfaces. Therefore we need a solution that can solve this problem, even by a little bit.

# Why a new algorithm?

If you are a common pleb, you would just do something like this:

```c
#include <math.h>

float q_rsqrt(float number)
{
    return 1/sqrt(x);
}
```

The ALU (Arithmetic and logic unit) in our CPUs have specialized system of gates built for addition and multiplication, but not for division and subtraction. Although subtraction has a quick fix with [2's complement method](https://www.geeksforgeeks.org/subtraction-of-two-numbers-using-2s-complement/), Division is still a more complex calculation, even more complex is square root. Intel and other manufacturers could, in theory add specialized gates for them, but it is very expensive. For thousands of calculations a second(like in our case), these gates arw simply not feasable. So we need something that can be faster. The fast inverse square root is an approximation with an error of atmost 1%, while being about 3x as fast.

# Floating point numbers
Before we proceed any further, I think it is required to mention how computers store floats. This is going to be very boring, so there's a TLDR after this.

Floats are expressed in a similar fashion to scientific notations; called the [IEEE-754 standard](https://en.wikipedia.org/wiki/IEEE_754) to put it simply:

- $42069$ in scientific notation is $4.2069 \times 10^3$ \
- $0.0042069$ in scientific notation is $4.2069 \times 10^{-3}$ \
- $42069$ in binary is $1010010001010101$ or $1.010010001010101 \times 2^{15}$ \

The smart folks at IEEE set it in this form\

![images](/images/Untitled-1.svg)
- Where the sign bit is 1 when the number is negative, 0 when positive. Since we'll be dealing exclusively with positive numbers with this algorithm (we would never need to calculate $\frac{1}{\sqrt{-5}}$ or something) the sign bit will always be 0.\



- 8 bits of Expoenents in Excess-127 format, you can represent exponents from 0 to 255 but we need to represent fractions (ie, negative exponents) too, so we shift the explonents by 127 so now we can represent explonents from -127 to 128 so the power 4, instead of 00000100, will be represented as 10000011 (this is where the term excess-127 comes from because the exponent is added by 127).

- The remaining 23 bits store the fractional part, from 1.0000000.... to 1.11111.... so [1, 2). But IEEE realized that the first digit will always be 1, so they made it part of the equation. Therefore the 23 bits now store only the Mantissa. (The part after the decimal point).\

The thing we discussed just now is what's called normalized numbers; IEEE also specifies denormalised numbers, NaN (Not a number) and two zeros (0 and -0). but this algorithm will never take those inputs and therefore won't be discussed here.

## TLDR

The number that we recieve will have 32 bits, first one being 0, next 8 being the exponent(E) and the remaining 23 being the Mantissa(M).

written in the form of ${2^{23}}*E + M$.\

$M$ = 01001110010000000000000\

$E$ = 10001001\

${2^{23}}*E$ = 10001001 00000000000000000000000

${2^{23}}*E + M$ = 10001001 01001110010000000000000\

# Bits and Numbers

From our Floating point thing, our number can be represented as:

$$n = \left(1+ \frac{M}{2^{23}}\right)*2^{E-127}$$

Taking $log_2$ on both sides:

$$log_2n = log_2\left(\left(1+ \frac{M}{2^{23}}\right)*2^{E-127}\right)$$

Simplifying:

$$log_2n = log_2\left(1+ \frac{M}{2^{23}}\right)+2^{E-127}$$

Using $log_2(1+x) \approx x$ ([source](https://math.stackexchange.com/a/1111064)):

$$log_2n = \frac{M}{2^{23}}+E+\mu-127$$ 

Note: the $\mu$ value is 0.0430, this value shifts the approximation to reduce the average error when $0 \leq x \leq 1$.\

Multiplying and dividing by $2^{23}$

$$log_2n = \frac{1}{2^{23}}\left(M+2^{23}*E\right)+\mu-127$$

Now we see that we just got the bit representation of our number. Therefore, the log of our number stores the bit representiaton, abeit with some scaling and shifting.

# The Steps

## Iterating through the code
Looking at the code again, the first 4 lines don't seem that harmful.

```c
float q_rsqrt(float number){         > 32 bit decimal number (input)
  long i;                            > 32-bit integer number
  float x2, y;                       > 32-bit decimal numbers
  const float threehalfs = 1.5F;     > 3/2, also 32-bit.
```
The next two aren't that bad either:

```c
  x2 = number * 0.5F;                > Assign number/2 to x2
  y  = number;                       > Assign number to y
```
But then, all hell seems to break loose, what is _i_? What is 0x5f3759df? Why declare a variable for 1.5 and not for 0x5f3759df? Why are there so many pointers?\

And the comments don't seem to help either. But it hints that there's three steps in the process, namely:

- evil floating point bit hack
- what the fuck
- 1st iteration (spoiler: Newton's approximations)

## Evil floating point bit hack

```c
i = *( long * ) &y;
```

Now the problem with floats in C is that it doesn't support [bit manipulations](https://en.wikipedia.org/wiki/Bit_manipulation).\

But longs do.\

So this line hacks C into thinking that the content stored in `y`'s address is actually a long, and then stores the data into i, which is actually a long.\

`&y` $\Rightarrow$ get the memory address of y.\
`(long * )&y` $\Rightarrow$ turn the data stored in the address y into a long.\
`i = *( long * ) &y;` $\Rightarrow$ store that data in i.

Now we know that both floats and longs have 32 bit memory allocations, So this line creates a one-to-one mapping of the bits from `y` to `i`.
Now our number can go through bit manipulations.

## What the fuck?

```c
i  = 0x5f3759df - ( i >> 1 ); 
```

Let's talk about bit manipulation, namely shifting. In binary, shifting left doubles the number and shifting right halves it, while rounding it off.

- x = 1101 = 13
- (x << 1) = 11010 = 26
- (x >> 1) = 110 = 6

Lets see how bit shifting affects exponents:\

- let our exponent be $n^x$.
- left shifting a exponent doubles it. ie: $n^{2x}$
- right shifting a exponent gives us the square root. ie: $n^{\frac{x}{2}}$

We have our number $y$ and we have to find out $\frac{1}{\sqrt{y}}$

But as we've seen:

$$log_2(y) \approx i$$

so let's just calculate 

$$log_2\left(\frac{1}{\sqrt{y}}\right)$$

which is equal to:

$$-\frac{1}{2}log_2(y)$$

Calculating this is stupidly easy. "But you just told me that division is difficult!!1!!" Yes but remember bit shifting???\
Just do `-(i >> 1)` and you're all set.

Now you might be wondering how and why do we have `0x5f3759df` in the line. Go to the end of [this](http://mayukh-chr.github.io/posts/draft-fast-inverse-square-root-algorithm/#bits-and-numbers) and read "abeit with some scaling and shifting.". Meaning that we need to scale and shift it back by some constant.

Let $$log(\Gamma) = log\left(\frac{1}{\sqrt{y}}\right)$$

which equals to

$$log(\Gamma) = -\frac{1}{2}log_2(y)$$

Now we replace the logarithm with the bit representation

$$\frac{1}{2^{23}}\left(M_\Gamma + 2^{23}*E_\Gamma\right) + \mu -127 = -\frac{1}{2}\left(\frac{1}{2^{23}}(M_\Gamma + 2^{23}*E_\Gamma) + \mu -127\right)$$

Calculating for $\left(M_\Gamma + 2^{23}*E_\Gamma\right)$

$$\left(M_\Gamma + 2^{23}*E_\Gamma\right) = \frac{3}{2}2^{23}\left(127-\mu\right)- \frac{1}{2} \left(M_y + 2^{23}*E_y\right)$$\


where: $\frac{3}{2}2^{23}\left(127-0.0430\right) \approx$ 0x5F3759DF.\

Therefore:

= 0x5F3759DF - (i >> 1)   (Note: Much later a more accurate constant was derived: 0x5F375A86, but we'll ignore that for now.)

```c
y  = * ( float * ) &i;
```

This is just reversing the steps of the evil bit hack to get back the actual approximation of those bits.
## First Iteration


```c
y  = y * ( threehalfs - ( x2 * y * y ) );
```

After the previous step, we have a pretty good approximation but we did pick up some error terms here and there, but we can use [newtown's approximation](https://en.wikipedia.org/wiki/Newton%27s_method) to get a really good approximation.

Newton's method finds a root of an equation. ie it finds an $x$ for which $f(x) = 0$. You repeat this process until you're satisfied with your solution. But in this case, our initial approximation is good enough that one iteration of it gets our error to $\leq1%$.\

The only thing that newton's method needs is a function and it's derivative. To put it simply:

$$x_{new} = x - \frac{f(x)}{f\prime(x)}$$

here it's: 

$$f(y) = 0 = \frac{1}{y^2}-x$$

$$\because f(y) = 0$$
$$\therefore y = \frac{1}{\sqrt{x}}$$

therefore using newton's method:

$$f(y) = \frac{1}{y^2}-x and  f\prime(y) =  -\frac{2}{y^3}$$

$$y_{n+1} = y_n - \frac{f(y_n)}{f\prime(y_n)}$$
which is equal to 
$$y_{n+1} = \frac{y_n\left(3-xy_n^2\right)}{2} $$

which is the last line.

```c
return y;
}
```

just return the result that we just got. and you're done.

# Subsequent Improvements

It is not known precisely how the exact value for the magic number was determined. But many speculate it's been through trial and error. Chris Lomont developed a function to minimize approximation error by choosing the magic number R R over a range. He first computed the optimal constant for the linear approximation step as 0x5F37642F, close to 0x5F3759DF, but this new constant gave slightly less accuracy after one iteration of Newton's method. Lomont then searched for a constant optimal even after one and two Newton iterations and found 0x5F375A86, which is more accurate than the original at every iteration stage.\

Subsequent additions by hardware manufacturers have made this algorithm redundant for the most part. For example, on x86, Intel introduced the SSE instruction rsqrtss in 1999. In a 2009 benchmark on the Intel Core 2, this instruction took 0.85ns per float compared to 3.54ns for the fast inverse square root algorithm, and had less error.