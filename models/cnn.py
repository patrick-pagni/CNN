import torch
from torch import nn

class IntermediateBlock(nn.Module):

  """
  Class for the intermediate convolutional layer with architecture specifed by
  assignment.
  """

  def __init__(
      self,
      input_channels = 3,
      output_channels = 256,
      output_volume = 32,
      kernel_size = 8,
      units = 4,
      groups = True
      ):
    """
    input_channels:
      The number of channels for incoming image
    output_channels:
      The number of channels for outgoing image
    output_volume:
      The size of the output image
    units:
      The number of convolutional units within the block
    groups:
      Whether to split inputs to groups for convolving
    """

    super(IntermediateBlock, self).__init__()

    self.units = units

    if not groups:
      self.groups = 1
    else:
      self.groups = input_channels

    # Set up the feed-forward network with the same number of outputs as units
    # for weighting each convolutional unit within block
    self.fc = nn.Sequential(
        nn.LayerNorm(input_channels),
        nn.Linear(input_channels, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, self.units),
        nn.Softmax(dim = 1)
    )

    # Each convolutional unit applies GELU activation function to outputs
    # Adaptive Average Pooling is used to ensure output dimensionality is
    # consistent
    self.conv_layer = nn.ModuleList([
        nn.Sequential(
          nn.Conv2d(
              input_channels,
              output_channels,
              groups = self.groups,
              kernel_size = kernel_size,
              stride = 1,
              padding = "same"
              ),
          nn.GELU(),
          nn.AdaptiveAvgPool2d(output_volume)
        )
    ] * units)

  def forward(self, x):

    # Calculate the mean of input x and pass it through a feed forward network
    # This returns weights specifying which convolutional units to weigh more
    m = torch.mean(x, dim = [2, 3])
    a = self.fc(m)

    # Calculate the output of each convolutional layer and multiply it by
    # the corresponding coefficient generated by the feed forward network above
    # Stack and sum outputs of each convolutional layer
    conv_layer_out = []

    for coef, unit in zip(a.T, self.conv_layer):
      conv_out = unit(x)
      coef = coef.reshape(-1,1,1,1)
      conv_layer_out.append(torch.mul(coef, conv_out))

    return torch.sum(torch.stack(conv_layer_out,dim = 0), dim = 0)

class OutputBlock(nn.Module):

  """
  Class for the output bloc with architecture specifed by assignment.
  """

  def __init__(self, input_channels, input_volume):

    super(OutputBlock, self).__init__()

    self.input_channels = input_channels
    self.input_volume = input_volume

    # Calculate size of flattened image for input layer to feed forward network
    self.first_layer_inputs = (input_volume * input_volume * input_channels)

    # Set up feed forward network
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.LayerNorm(self.first_layer_inputs),
      nn.Linear(self.first_layer_inputs, 10)
  )

  def forward(self, x):

    # Calculate logits from feed forward network
    logits = self.fc(x)

    return logits

class Residual(nn.Module):
  """
  Wrapper class which pools the original input and adds it to the output
  of the wrapped module.

  https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/models/convmixer.py
  """
  def __init__(self, fn):
    """
    fn:
      The module being wrapped
    """
    super().__init__()
    self.fn = fn

  def forward(self, x):

    # Calculate the output from the forward pass of the wrapped module
    output = self.fn(x)

    # Change the dimensionality of x to match that of the output
    residual = nn.AdaptiveMaxPool2d(output.shape[2])(x)

    # Sum the output with the residual
    return output + residual

class AdvancedCNN(nn.Module):

  """
  Class defining advanced convolutional architecture based on that specified in
  assignment
  """

  def __init__(self, block_config):

      """
      block_config:
        Dictionary containing the configuration for each intermediate block
      """

      super(AdvancedCNN, self).__init__()

      # Set up empty sequential container to iteratively add blocks based on
      # config dictionary
      self.spine = nn.Sequential()

      # Variable to track last intermediate block's output channels
      self.int_block_out = None

      # Add first intermediate block to expand channels from 3 to 256
      self.spine.append(nn.Sequential(
        IntermediateBlock(
            input_channels = 3,
            output_channels = block_config["block_1"]["input_channels"],
            output_volume = 32,
            units = 4,
            groups = False
        ),
        nn.BatchNorm2d(block_config["block_1"]["input_channels"])
        ))

      # Iterate through config dictionary and add each block to spine
      for i, block in enumerate(block_config.values()):

        self.int_block_out = block["output_channels"]

        # Add intermediate block wrapped with the residual class
        self.spine.append(
            nn.Sequential(
              Residual(
                  IntermediateBlock(**block)
                  ),
              nn.BatchNorm2d(self.int_block_out)
              )
            )

        # Add convolutional layer wrapped with residual layer
        self.spine.append(
            nn.Sequential(
                nn.Conv2d(
                    self.int_block_out,
                    self.int_block_out,
                    kernel_size=1
                    ),
                nn.GELU(),
                nn.BatchNorm2d(self.int_block_out)
                )
            )

      # Add adaptive pooling layer to reduce each channel to a number
      # representing one feature
      self.spine.append(nn.AdaptiveAvgPool2d((1,1)))

      # Add output block
      self.spine.append(OutputBlock(self.int_block_out, 1))

  def forward(self, x):

    return self.spine(x)