sound-rnn
=========

Generating sound using recurrent neural networks.

For more details, see my blog post: http://www.johnglover.net/blog/generating-sound-with-rnns.html


## Requirements

This code is written in Lua and requires [Torch](http://torch.ch/).
See the Torch installation documentation for more details.

The following packages are also required (and can be installed using [luarocks](https://luarocks.org)):

```bash
$ luarocks install nngraph
$ luarocks install optim
$ luarocks install nn
```

Also install the following (see the respective repositories for installation instructions):

* [torch-signal](https://github.com/soumith/torch-signal)
* [lua audio](https://github.com/soumith/lua---audio)


## Usage

Run `train.lua` with no parameters to get a list of arguments. The audio file that you pass in should be a single-channel wav file. For example (using the included sine tone):

```bash
$ th train.lua -audio audio/sine_440.wav -model_file sine_440.t7 -batch_size 10 -seq_length 10 -rnn_size 100 -num_partials 10 -mdn_components 1 -num_layers 1
```

To generate audio, run `sample.lua` (again no parameters for an argument list). For example:

```bash
$ th sample.lua -model sine_440.t7 -length 3000 -output sine_resynth.wav
```

## Thanks

This code is based on a number of great examples:

* [char-rnn](https://github.com/karpathy/char-rnn) by Andrej Karpathy
* Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6)
* [learning to execute](https://github.com/wojciechz/learning_to_execute) by Wojciech Zaremba

## License

MIT
