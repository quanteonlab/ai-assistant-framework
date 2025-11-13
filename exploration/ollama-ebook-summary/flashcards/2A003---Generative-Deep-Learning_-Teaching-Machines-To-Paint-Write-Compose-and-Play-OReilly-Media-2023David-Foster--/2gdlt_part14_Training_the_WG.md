# Flashcards: 2A003---Generative-Deep-Learning_-Teaching-Machines-To-Paint-Write-Compose-and-Play-OReilly-Media-2023David-Foster--_processed (Part 14)

**Starting Chapter:** Training the WGAN-GP

---

#### L2 Norm and Wasserstein Loss Calculation
Background context: The document explains how to calculate the L2 norm of a vector, which is used to measure the average squared distance from 1. This concept is foundational for understanding the next part of WGAN-GP training.

:p What is the relationship between the L2 norm and the average squared distance in this context?
??x
The L2 norm of a vector measures the Euclidean distance of the vector from the origin, which can be used to calculate the average squared distance from 1. However, it's not directly used in WGAN-GP but serves as an introductory concept for understanding distances.

This is relevant because understanding basic norms helps grasp more complex concepts like the Wasserstein loss.
x??

---

#### Training Step of WGAN-GP
Background context: The training step involves updating both the critic and generator networks iteratively. This process includes calculating various losses and gradients to improve the model's performance.

:p How does one train the WGAN-GP according to the provided code snippet?
??x
The training step for WGAN-GP involves multiple steps:
1. Training the critic three times.
2. Calculating the Wasserstein loss (c_wass_loss) by comparing predictions of real and fake images.
3. Adding a gradient penalty term to ensure smooth gradients.
4. Updating the critic's weights based on these losses.
5. Updating the generator’s weights using the negative average prediction from the critic.

Here is an excerpt of the code:

```python
def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]
    for i in range(3):  # Critic updates before generator update
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.critic(fake_images, training=True)
            real_predictions = self.critic(real_images, training=True)
            
            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
            
            # Gradient Penalty
            c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
            
            c_loss = c_wass_loss + c_gp * self.gp_weight

        c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))
    
    random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    
    with tf.GradientTape() as tape:
        fake_images = self.generator(random_latent_vectors, training=True)
        fake_predictions = self.critic(fake_images, training=True)

        g_loss = -tf.reduce_mean(fake_predictions)

    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

    self.c_loss_metric.update_state(c_loss)
    self.c_wass_loss_metric.update_state(c_wass_loss)
    self.c_gp_metric.update_state(c_gp)
    self.g_loss_metric.update_state(g_loss)

    return {m.name: m.result() for m in self.metrics}
```

x??

---

#### Critic Loss Calculation
Background context: The critic loss function combines the Wasserstein loss and a gradient penalty term. This ensures that the gradients are accurate and smooth, which is crucial for generator updates.

:p What does the critic loss calculation involve?
??x
The critic loss calculation involves two main components:
1. **Wasserstein Loss**: This measures the difference between the average prediction of real images and fake images.
2. **Gradient Penalty**: This ensures that the gradients are smooth, avoiding issues like vanishing or exploding gradients.

Here's how it is calculated in the provided code:

```python
c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
c_loss = c_wass_loss + c_gp * self.gp_weight
```

This combined loss ensures that the critic provides accurate and smooth gradients for generator updates.

x??

---

#### Generator Loss Calculation
Background context: The generator's goal is to fool the critic by producing images that are as realistic as possible. Its loss function is derived from the critic’s predictions, aiming to maximize the similarity between real and fake image predictions.

:p How does one calculate the generator loss in WGAN-GP?
??x
The generator loss is calculated based on the negative average prediction from the critic, which encourages the generator to produce images that are considered realistic by the critic. The formula used is:

```python
g_loss = -tf.reduce_mean(fake_predictions)
```

This negative mean value is aimed at maximizing the generator's ability to generate fake images that trick the critic into thinking they are real.

x??

---

#### Critic Gradient Penalty Calculation
Background context: The gradient penalty term ensures that the critic's decision boundaries remain smooth, which is important for accurate and stable training. This term penalizes deviations from a straight line in the parameter space.

:p What is the purpose of calculating a gradient penalty in WGAN-GP?
??x
The purpose of calculating a gradient penalty in WGAN-GP is to ensure that the critic's decision boundaries remain smooth, avoiding sharp discontinuities that could lead to unstable training. This term penalizes deviations from straight lines in the parameter space, ensuring that gradients are consistently informative.

Here’s how it might be calculated:

```python
c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
```

x??

---

#### Critic Training and Generator Updates
Background context: In WGAN-GP, critics are trained several times between each generator update. This ensures that the critic is close to convergence before updating the generator, providing accurate gradients for further training.

:p How many times should the critic be updated relative to the generator in WGAN-GP?
??x
In WGAN-GP, it is typical to train the critic three to five times between each generator update. This ensures that the critic is close to convergence and provides more accurate gradients for the generator updates.

Example ratio: 3-5 critic updates per generator update.
x??

---

#### Batch Normalization in a WGAN-GP
Background context: Batch normalization should not be used in the critic of a WGAN-GP because it creates correlations between images within a batch, making the gradient penalty less effective. However, experiments have shown that removing batch normalization can still lead to excellent results.

:p Why is batch normalization not recommended for use in the critic of a WGAN-GP?
??x
Batch normalization should not be used in the critic of a WGAN-GP because it introduces correlations between images within a single batch, which can make the gradient penalty less effective. This correlation can disrupt the smoothness and stability required for accurate training.

Removing batch normalization from the critic has been shown to still produce excellent results in practice.
x??

---

#### WGAN-GP Generator and Critic Loss Functions
Background context: The Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) uses a different loss function compared to standard GANs. In WGAN-GP, the generator and critic use a different approach to optimize their objectives.
The objective of the critic in WGAN-GP is to minimize the Wasserstein distance between real and fake data distributions.

:p What are the main differences in the loss functions used by the generator and critic in WGAN-GP compared to standard GANs?
??x
In WGAN-GP, both the generator and critic aim to optimize their objectives differently:
- The generator aims to maximize the Wasserstein distance, while the critic aims to minimize it.
- Unlike in standard GANs where a discriminator outputs probabilities close to 1 for real data and 0 for fake data, the critic in WGAN-GP outputs values that represent the estimated Wasserstein distance between distributions.

The loss functions for both are:
- Critic: Minimize $\mathbb{E}_{\boldsymbol{x} \sim p_{data}} [f(\boldsymbol{x})] - \mathbb{E}_{\boldsymbol{z} \sim p_z} [f(G(\boldsymbol{z}))] + \lambda \cdot \text{GP}(G, f)$- Generator: Maximize $-\mathbb{E}_{\boldsymbol{z} \sim p_z} [f(G(\boldsymbol{z}))]$

Where:
- $f$ is the critic's output
- $G $ generates fake data from latent variables$\boldsymbol{z}$-$\lambda \cdot \text{GP}(G, f)$ is the gradient penalty term.

No specific code examples here as it is more about understanding the logic and formulas.
x??

---

#### Gradient Penalty Term in WGAN-GP
Background context: To ensure that the critic's output values are meaningful in terms of representing distances between distributions, a gradient penalty term is added to the loss function. This helps in ensuring that the critic behaves like a good discriminator by making its decision boundaries smooth.

:p What is the role of the gradient penalty term in WGAN-GP?
??x
The role of the gradient penalty term in WGAN-GP is to ensure that the gradients of the critic with respect to the input images are close to 1, which helps in making the critic's output values meaningful. This smoothness constraint ensures that small perturbations in the input space result in small changes in the critic’s outputs.

The gradient penalty term is calculated as follows:
$$\text{GP} = \mathbb{E}_{\boldsymbol{\epsilon} \sim U(0,1)} \left[ \| \nabla_{\boldsymbol{x}} \left(L(\boldsymbol{x} + \boldsymbol{\epsilon} (\hat{\boldsymbol{x}} - \boldsymbol{x})) \right) \|^2_2 - 1 \right]^2$$where $ L $ is the critic's loss, and $\boldsymbol{x}$ and $\hat{\boldsymbol{x}}$ are real and fake data points respectively.

The gradient penalty term is added to the critic’s loss:
$$c_{\text{loss}} = \mathbb{E}_{\boldsymbol{x} \sim p_{data}} [f(\boldsymbol{x})] - \mathbb{E}_{\boldsymbol{z} \sim p_z} [f(G(\boldsymbol{z}))] + \lambda \cdot \text{GP}(G, f)$$x??

---

#### Conditional GAN (CGAN)
Background context: While standard GANs generate images based on a random latent vector, CGANs allow for additional conditioning information to be incorporated. This is useful in scenarios where we want to control the attributes of generated images.

:p What is the main difference between a standard GAN and a Conditional GAN (CGAN)?
??x
The main difference between a standard GAN and a Conditional GAN (CGAN) lies in how they incorporate additional information during the generation process:

- **Standard GAN**: Generates images from a random latent vector without any explicit control over image attributes.
- **Conditional GAN (CGAN)**: Incorporates an additional one-hot encoded label as input to both the generator and critic, allowing for more controlled generation.

For example, if you want to generate faces with specific hair colors, in a CGAN, this attribute is provided as a label during training. This helps ensure that generated images match the specified attributes.

Example architecture:
- **Generator Input**: Latent vector $\boldsymbol{z}$ and one-hot encoded label vector.
- **Critic Input**: Image input and corresponding one-hot encoded label.

This enables more controlled generation where we can explicitly specify what kind of image to generate based on provided labels.

x??

---

#### CGAN Generator and Critic Architectures
Background context: In a Conditional GAN (CGAN), the generator and critic architectures are extended to include an additional input for condition labels. This allows for generating images with specific attributes controlled by these labels.

:p How do the architecture changes in CGANs affect the inputs to the generator and critic?
??x
In CGANs, both the generator and critic receive additional information (labels) as part of their inputs:

- **Generator**: Receives a latent vector $\boldsymbol{z}$ and a one-hot encoded label vector. The generator concatenates these two vectors before processing them.
- **Critic**: Receives an image input and a one-hot encoded label vector. Similar to the generator, it concatenates both inputs.

For example:
```python
# Generator Input Layers
generator_input = layers.Input(shape=(32,))
label_input = layers.Input(shape=(2,))
x = layers.Concatenate(axis=-1)([generator_input, label_input])
x = layers.Reshape((1, 1, 34))(x)

# Critic Input Layers
critic_input = layers.Input(shape=(64, 64, 3))
label_input = layers.Input(shape=(2,))
x = layers.Concatenate(axis=-1)([critic_input, label_input])
```

These changes ensure that the generator and critic have access to both image data and condition labels, allowing for more controlled generation of images.

x??

---

#### Training CGAN
Background context: Training a Conditional GAN (CGAN) requires adapting the training process to account for the additional conditioning information. This involves modifying the `train_step` function to handle the new input formats.

:p How does the training loop for a CGAN differ from that of a standard GAN?
??x
Training a Conditional GAN (CGAN) differs primarily in how it handles the generator and critic updates, especially due to the additional conditioning information:

1. **Generator**: Receives both a latent vector and a label as inputs.
2. **Critic**: Receives an image input and corresponding label.

The `train_step` function for CGAN is modified to accommodate these changes:
```python
def train_step(self, data):
    real_images, one_hot_labels = data
    image_one_hot_labels = one_hot_labels[:, None, None, :]
    image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=64, axis=1)
    image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=64, axis=2)

    batch_size = tf.shape(real_images)[0]
    for i in range(self.critic_steps):
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, one_hot_labels], training=True)
            fake_predictions = self.critic([fake_images, image_one_hot_labels], training=True)
            real_predictions = self.critic([real_images, image_one_hot_labels], training=True)

            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
            c_gp = self.gradient_penalty(batch_size, real_images, fake_images, image_one_hot_labels)
            c_loss = c_wass_loss + c_gp * self.gp_weight
            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

    random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    with tf.GradientTape() as tape:
        fake_images = self.generator([random_latent_vectors, one_hot_labels], training=True)
        fake_predictions = self.critic([fake_images, image_one_hot_labels], training=True)
        g_loss = -tf.reduce_mean(fake_predictions)
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

```

This code ensures that the generator and critic are updated appropriately during each step of training.

x??

---

#### DCGAN Training for Toy Brick Images
Background context explaining how a DCGAN is trained to generate images of toy bricks, including the architecture and training process. The DCGAN learns to represent 3D objects realistically, capturing attributes like shadow, shape, and texture.

:p How does a DCGAN train to generate realistic images of toy bricks?
??x
The DCGAN trains by having a generator that generates toy brick images from random latent vectors, while the discriminator aims to distinguish between real toy brick images and fake ones generated by the generator. The generator tries to fool the discriminator.

```python
# Pseudocode for training DCGAN
def train_dcgan(discriminator, generator, dataset):
    for epoch in range(num_epochs):
        # Train Discriminator
        real_images = get_real_images_from_dataset()
        fake_images = generator.predict(latent_vectors)
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        g_loss = generator.train_on_batch(latent_vectors, real_labels)
```
x??

---

#### Wasserstein GAN with Gradient Penalty (WGAN-GP) Training
Background context explaining the WGAN-GP model and its improvement over standard GANs by addressing issues like mode collapse and vanishing gradients. The key feature is the 1-Lipschitz constraint on the critic, enforced through a gradient penalty.

:p How does the WGAN-GP improve upon traditional GANs?
??x
The WGAN-GP improves upon traditional GANs by imposing a 1-Lipschitz constraint on the critic (discriminator), ensuring that the gradient of the critic's output with respect to its input remains close to 1. This is achieved through an additional penalty term in the loss function.

```python
# Pseudocode for WGAN-GP
def train_wgan_gp(critic, generator, dataset):
    # Train Critic
    real_images = get_real_images_from_dataset()
    fake_images = generator.predict(latent_vectors)
    
    d_loss_fake = critic.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_loss_real = critic.train_on_batch(real_images, np.ones((batch_size, 1)))
    gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images)
    d_loss = -0.5 * (d_loss_fake + d_loss_real) + 10 * gradient_penalty
    
    # Train Generator
    g_loss = generator.train_on_batch(latent_vectors, np.ones((batch_size, 1)))
```
x??

---

#### Conditional GAN (CGAN)
Background context explaining how a CGAN uses labels to condition the generated images. The CGAN can control specific attributes in the generated output by conditioning on certain label vectors.

:p How does a CGAN use conditional inputs to generate images?
??x
A CGAN incorporates conditional information, such as labels, into both the generator and discriminator (referred to as a critic). During training, the generator is fed with random latent vectors along with one-hot encoded label vectors. The critic evaluates the generated images based on these labels, helping the generator learn to produce outputs that match specific attributes.

```python
# Pseudocode for CGAN Training
def train_cgan(generator, critic, dataset):
    # Train Critic
    real_images, labels = get_real_samples_and_labels(dataset)
    fake_images = generator.predict([latent_vectors, labels])
    
    d_loss_fake = critic.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_loss_real = critic.train_on_batch(real_images, np.ones((batch_size, 1)))
    gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images)
    d_loss = -0.5 * (d_loss_fake + d_loss_real) + 10 * gradient_penalty
    
    # Train Generator
    g_loss = generator.train_on_batch([latent_vectors, labels], np.ones((batch_size, 1)))
```
x??

---

#### Generator Training in CGAN
Background context on how the generator in a CGAN is trained to produce images that match specific attributes defined by the conditional inputs.

:p How does the generator in a CGAN learn to generate images based on conditional inputs?
??x
The generator in a CGAN learns to generate images conditioned on specific labels. During training, it receives random latent vectors and one-hot encoded label vectors as input. The goal is for the generator to produce images that match the attributes specified by these labels.

```python
# Pseudocode for CGAN Generator Training
def train_generator(generator, critic, dataset):
    # Train Generator
    latent_vectors = generate_random_latent_vectors()
    labels = one_hot_encode_labels()  # e.g., [1, 0] or [0, 1]
    
    g_loss = generator.train_on_batch([latent_vectors, labels], np.ones((batch_size, 1)))
```
x??

---

#### Critic Training in CGAN
Background context on how the critic in a CGAN evaluates images based on both their appearance and conditional information.

:p How does the critic in a CGAN work during training?
??x
The critic in a CGAN evaluates the quality of generated images by considering both their visual appearance and the labels they are supposed to match. During training, it receives fake and real images along with corresponding one-hot encoded label vectors. The goal is for the critic to distinguish between real and fake images based on these combined inputs.

```python
# Pseudocode for CGAN Critic Training
def train_critic(critic, generator, dataset):
    # Train Critic
    real_images, labels = get_real_samples_and_labels(dataset)
    fake_images = generator.predict([latent_vectors, labels])
    
    d_loss_fake = critic.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_loss_real = critic.train_on_batch(real_images, np.ones((batch_size, 1)))
    gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images)
    d_loss = -0.5 * (d_loss_fake + d_loss_real) + 10 * gradient_penalty
```
x??

---

#### Gradient Penalty Computation in WGAN-GP
Background context on the importance of maintaining a 1-Lipschitz constraint in the critic to ensure stable training.

:p How is the gradient penalty computed in WGAN-GP?
??x
The gradient penalty in WGAN-GP measures how far the gradients of the critic's output with respect to its input deviate from being exactly 1. This ensures that the critic remains 1-Lipschitz, which helps stabilize training.

```python
# Pseudocode for Computing Gradient Penalty
def compute_gradient_penalty(critic, real_images, fake_images):
    # Interpolate between real and fake images
    alpha = np.random.uniform(0., 1., size=(real_images.shape[0], 1, 1, 1))
    interpolates = (alpha * real_images + (1 - alpha) * fake_images)
    
    # Get gradients w.r.t. interpolates
    grads = tf.GradientTape().gradient(critic(interpolates), [interpolates])
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((norm - 1.) ** 2)
```
x??

---

#### Generating Images with CGAN
Background context on how to use a CGAN to generate images that match specific attributes defined by the conditional inputs.

:p How do you use a CGAN to generate an image of a nonblond face?
??x
To generate an image of a nonblond face using a CGAN, you would pass in the appropriate label vector [0, 1] representing "not blond" along with random latent vectors. The generator will produce an image where the hair color is controlled by this label.

```python
# Pseudocode for Generating Nonblond Face with CGAN
def generate_nonblond_face(generator, latent_vectors):
    labels = one_hot_encode_label("not_blonde")  # e.g., [0, 1]
    generated_image = generator.predict([latent_vectors, labels])
```
x??

---

#### Evaluating GAN Performance
Background context on common issues in GAN training such as mode collapse and vanishing gradients.

:p What are the main challenges in training GANs like DCGAN, WGAN-GP, and CGAN?
??x
Common challenges in training GANs include:
- **Mode Collapse**: The generator learns to produce only a few modes (types of images) rather than capturing the full diversity.
- **Vanishing Gradients**: In deep networks, gradients can vanish during backpropagation, hindering learning.

To address these, methods like WGAN-GP use gradient penalties and 1-Lipschitz constraints to stabilize training. CGANs condition on labels to better control attribute generation.

```python
# Pseudocode for Addressing Mode Collapse in GAN Training
def train_gan_model(model):
    # Training loop
    for epoch in range(num_epochs):
        real_images = get_real_samples_from_dataset()
        fake_images, labels = generate_fake_samples(generator, latent_vectors)
        
        critic_loss = train_critic(critic, real_images, fake_images, labels)
        generator_loss = train_generator(generator, labels)
```
x??

---

