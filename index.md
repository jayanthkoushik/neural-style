---
title: neural-style
layout: default
---

This page contains pre-trained models and examples from
[my implementation](https://github.com/jayanthkoushik/neural-style)
of style transfer algorithms. Everything here is based on the method described
by Justin Johnson, Alexandre Alahi, and Fei-Fei Li in
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).
My implementation has some modifications, the details of which can be found on
the GitHub page. There, you can also find instructions for training models, and
using them to style images.

With a trained model, you can process images super fast - all it takes is a
single pass through a convolutional neural network. Of course this needs a GPU
which means you can't use very large images due to the limited memory on GPUs.
You *can* style much larger images on a CPU but it will take a lot of time and
memory. Styling this photo (4032x3024) took more than 5 minutes, and about 120
gigabytes of memory; but it makes for a pretty cool wallpaper.

{% include img_gallery.html source="xl" %}

The remaining images on this page are not so huge, and were processed in about
100 milliseconds. The original images are shown below.

{% include img_gallery.html source="original" %}

Next are these images modified by various styles. For each style, the
trained model file is provided, and the first image in each section is the style
image used for the model. Unless specified otherwise, default values were used
for training arguments.

# Stained Glass
{% include img_display.html source="style/fulls/stained_glass.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/stained_glass.h5)]<br>
For this model, the style weight was set to 0.0001.
{% include img_gallery.html source="stained_glass" %}

# Composition X <small>Wassily Kandinsky (1939)</small>
{% include img_display.html source="style/fulls/kandinsky_x.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/kandinsky_x.h5)]<br>
For this model, the style size was set to 512.
{% include img_gallery.html source="kandinsky_x" %}

# Portrait de Jean Metzinger <small>Robert Delaunay (1906)</small>
{% include img_display.html source="style/fulls/metzinger.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/metzinger.h5)]
{% include img_gallery.html source="metzinger" %}

# Composition XIV <small>Piet Mondrian (1913)</small>
{% include img_display.html source="style/fulls/mondrian_xiv.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/mondrian_xiv.h5)]<br>
For this model, the content weight was set to 10.
{% include img_gallery.html source="mondrian_xiv" %}

# Udnie <small>Francis Picabia (1913)</small>
{% include img_display.html source="style/fulls/udnie.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/udnie.h5)]
{% include img_gallery.html source="udnie" %}

# Untitled, CR1091 <small>Jackson Pollock (1951)</small>
{% include img_display.html source="style/fulls/pollock_untitled.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/pollock_untitled.h5)]<br>
For this model, the content weight was set to 5, the style weight was set to
0.0001, and the style size was set to 400.
{% include img_gallery.html source="pollock_untitled" %}

# The Great Wave off Kanagawa <small>Hokusai (1830–1833)</small>
{% include img_display.html source="style/fulls/wave.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/wave.h5)]
{% include img_gallery.html source="wave" %}

# Cossacks <small>Wassily Kandinsky (1910)</small>
{% include img_display.html source="style/fulls/cossacks.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/cossacks.h5)]<br>
For this model, the content weight was set to 10.
{% include img_gallery.html source="cossacks" %}

# Flames
{% include img_display.html source="style/fulls/flames.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/flames.h5)]
{% include img_gallery.html source="flames" %}

# One: Number 31 <small>Jackson Pollock (1950)</small>
{% include img_display.html source="style/fulls/one31.jpg" %}
[[model](https://github.com/jayanthkoushik/neural-style-models/raw/master/one31.h5)]
{% include img_gallery.html source="one31" %}
