import keras
import keras.backend as K
import tensorflow as tf

import numpy as np
import glob, os
import argparse

import utils
import loggingreporter

import six
from six.moves import cPickle
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')

import kde
import simplebinmi

import utils 

ap = argparse.ArgumentParser()
ap.add_argument('-b', '--batch_size', help='Batch Size for every epoch', type=int, required=True)
ap.add_argument('-l', '--lr', help='Learning Rate', default=0.001, type=float)
ap.add_argument('-n', '--num_epochs', help='Max number of training epochs', type=int, required=True)
ap.add_argument('-a', '--activation', help='Activation function (tanh/relu)', required=True)
ap.add_argument('-w', '--weights_config', help='Configuration of sizes for weights in different layers (eg. 20-20-20-20-20)', required=True)
ap.add_argument('-s', '--start', help='Epoch number to begin from', type=int, required=True)
ap.add_argument('-g', '--num_gpu', help='Number of GPUs', type=int, default=1)
ap.add_argument('-p', '--plot', help='Plot Mutual Info plane using checkpoints created! No training!', action='store_true')
args = ap.parse_args()

layers = [int(p) for p in args.weights_config.split('-')]

trn, tst = utils.get_mnist()

if not os.path.isdir('rawdata'):
    os.mkdir('rawdata')
args.save_dir = os.path.join('rawdata',args.activation+'_'+args.weights_config)
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

if args.start==1:
    # A Simple Feed-Forward NN with given activation for 1D MNIST
    input_layer  = keras.layers.Input((trn.X.shape[1],))
    clayer = input_layer
    activate = keras.layers.advanced_activations.LeakyReLU(alpha=0.03) if args.activation=='leaky' else keras.layers.advanced_activations.PReLU()
    for n in layers:
        if args.activation in ['leaky', 'prelu']:
            clayer = keras.layers.Dense(n, activation='linear')(clayer)
            clayer = activate(clayer)
        else:
            clayer = keras.layers.Dense(n, activation=args.activation)(clayer)               
    output_layer = keras.layers.Dense(trn.nb_classes, activation='softmax')(clayer)

    with tf.device('/cpu:0'):
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
else:
    with tf.device('/cpu:0'):
        model = keras.models.load_model(os.path.join(args.save_dir,'chkpt.h5'))
optimizer = keras.optimizers.SGD(lr=args.lr)

if args.num_gpu>1:
    model = keras.utils.multi_gpu_model(model, gpus=args.num_gpu)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def do_report(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 20:       # Log for all first 20 epochs
        return True
    elif epoch < 100:    # Then for every 5th epoch
        return (epoch % 5 == 0)
    elif epoch < 200:    # Then every 10th
        return (epoch % 10 == 0)
    else:                # Then every 100th
        return (epoch % 1000 == 0)
    
if not os.path.isdir(os.path.join(args.save_dir,'tensorboard')):
    os.mkdir(os.path.join(args.save_dir,'tensorboard'))
    
reporter = loggingreporter.LoggingReporter(args=args, 
                                          trn=trn, 
                                          tst=tst, 
                                          do_save_func=do_report)
                                          
                               
saver = keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir,'chkpt.h5'), 
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=False, 
                                        save_weights_only=False, 
                                        mode='auto', 
                                        period=1000)

scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            factor=0.75, 
                                            patience=20, 
                                            verbose=0, 
                                            mode='min')

visualizer = keras.callbacks.TensorBoard(log_dir=os.path.join(args.save_dir,'tensorboard'), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False)

if not args.plot:
    r = model.fit(x=trn.X, y=trn.Y, 
                  verbose    = 2, 
                  batch_size = args.batch_size,
                  epochs     = args.num_epochs,
                  initial_epoch = args.start-1,
                  validation_data=(tst.X, tst.Y),
                  callbacks  = [reporter, saver, scheduler, visualizer])

# Which measure to plot
infoplane_measures = ['bin', 'upper', 'lower']

# Functions to return upper and lower bounds on entropy of layer activity
noise_variance = 1e-1                    # Added Gaussian noise variance
Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

# nats to bits conversion factor
nats2bits = 1.0/np.log(2) 

# Save indexes of tests data for each of the output classes
saved_labelixs = {}
for i in range(10):
    saved_labelixs[i] = tst.y == i

labelprobs = np.mean(tst.Y, axis=0)

PLOT_LAYERS    = None     # Which layers to plot.  If None, all saved layers are plotted 

# Data structure used to store results
measures = OrderedDict()
measures[args.activation] = {}

# Compute MI measures

if not os.path.exists(os.path.join(*['rawdata',args.activation+'_'+args.weights_config, 'infoplane_measures'])):
    for activation in measures.keys():
        cur_dir = os.path.join('rawdata', args.activation+'_'+args.weights_config)
        if not os.path.exists(cur_dir):
            print("Directory %s not found" % cur_dir)
            continue
            
        # Load files saved during each epoch, and compute MI measures of the activity in that epoch
        print('*** Doing %s ***' % cur_dir)
        for epochfile in sorted(os.listdir(cur_dir)):
            if not epochfile.startswith('epoch'):
                continue
                
            fname = cur_dir + "/" + epochfile
            with open(fname, 'rb') as f:
                d = cPickle.load(f)

            epoch = d['epoch']
            if epoch in measures[activation]: # Skip this epoch if its already been processed
                continue                      # this is a trick to allow us to rerun this cell multiple times)
                
            if epoch > args.num_epochs:
                continue

            print("Doing", fname)
            
            num_layers = len(d['data']['activity_tst'])

            if PLOT_LAYERS is None:
                PLOT_LAYERS = []
                for lndx in range(num_layers):
                    #if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
                    PLOT_LAYERS.append(lndx)
                    
            cepochdata = defaultdict(list)
            for lndx in range(num_layers):
                activity = d['data']['activity_tst'][lndx]

                # Compute marginal entropies
                h_upper = entropy_func_upper([activity,])[0]
                h_lower = entropy_func_lower([activity,])[0]
                    
                # Layer activity given input. This is simply the entropy of the Gaussian noise
                hM_given_X = kde.kde_condentropy(activity, noise_variance)

                # Compute conditional entropies of layer activity given output
                hM_given_Y_upper=0.
                for i in range(10):
                    hcond_upper = entropy_func_upper([activity[saved_labelixs[i],:],])[0]
                    hM_given_Y_upper += labelprobs[i] * hcond_upper
                    
                hM_given_Y_lower=0.
                for i in range(10):
                    hcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
                    hM_given_Y_lower += labelprobs[i] * hcond_lower
                
                cepochdata['MI_XM_upper'].append( nats2bits * (h_upper - hM_given_X) )
                cepochdata['MI_YM_upper'].append( nats2bits * (h_upper - hM_given_Y_upper) )
                cepochdata['H_M_upper'  ].append( nats2bits * h_upper )

                pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
                cepochdata['MI_XM_lower'].append( nats2bits * (h_lower - hM_given_X) )
                cepochdata['MI_YM_lower'].append( nats2bits * (h_lower - hM_given_Y_lower) )
                cepochdata['H_M_lower'  ].append( nats2bits * h_lower )
                pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

                binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, 0.5)
                cepochdata['MI_XM_bin'].append( nats2bits * binxm )
                cepochdata['MI_YM_bin'].append( nats2bits * binym )
                pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])
            
                print('- Layer %d %s' % (lndx, pstr) )

            measures[activation][epoch] = cepochdata

    with open(os.path.join(cur_dir, 'infoplane_measures'), 'wb') as f:
        cPickle.dump(measures, f, protocol=cPickle.HIGHEST_PROTOCOL)
else:
    with open(os.path.join(*['rawdata',args.activation+'_'+args.weights_config, 'infoplane_measures']), 'rb') as f:
        measures = cPickle.load(f)
    PLOT_LAYERS = [x for x in range(len(args.weights_config.split('-')))]
    
# Plot overall summaries

plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(4,2)
for actndx, (activation, vals) in enumerate(measures.items()):
    epochs = sorted(vals.keys())
    if not len(epochs):
        continue
        
    plt.subplot(gs[0,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        xmvalsU = np.array([vals[epoch]['H_M_upper'][layerid] for epoch in epochs])
        xmvalsL = np.array([vals[epoch]['H_M_lower'][layerid] for epoch in epochs])
        plt.plot(epochs, xmvalsU, label='Layer %d'%layerid)
        #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(activation)
    plt.ylabel('H(M)')
    
    plt.subplot(gs[1,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        #for epoch in epochs:
        #    print('her',epoch, measures[activation][epoch]['MI_XM_upper'])
        xmvalsU = np.array([vals[epoch]['MI_XM_upper'][layerid] for epoch in epochs])
        xmvalsL = np.array([vals[epoch]['MI_XM_lower'][layerid] for epoch in epochs])
        plt.plot(epochs, xmvalsU, label='Layer %d'%layerid)
        #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
    plt.xscale('log')
    plt.ylabel('I(X;M)')


    plt.subplot(gs[2,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        ymvalsU = np.array([vals[epoch]['MI_YM_upper'][layerid] for epoch in epochs])
        ymvalsL = np.array([vals[epoch]['MI_YM_lower'][layerid] for epoch in epochs])
        plt.plot(epochs, ymvalsU, label='Layer %d'%layerid)
    plt.xscale('log')
    plt.ylabel('MI(Y;M)')

    plt.subplot(gs[3,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        hbinnedvals = np.array([vals[epoch]['MI_XM_bin'][layerid] for epoch in epochs])
        plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
    plt.xlabel('Epoch')
    #plt.ylabel("H'(M)")
    plt.ylabel("I(X;M)bin")
    #plt.yscale('log')
    
    if actndx == 0:
        plt.legend(loc='lower right')
        
plt.tight_layout()


# Plot Infoplane Visualization

max_epoch = max( (max(vals.keys()) if len(vals) else 0) for vals in measures.values())
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=args.num_epochs))
sm._A = []

for infoplane_measure in infoplane_measures:
    fig=plt.figure(figsize=(10,5))
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue
        plt.subplot(1,2,actndx+1)    
        for epoch in epochs:
            c = sm.to_rgba(epoch)
            xmvals = np.array(vals[epoch]['MI_XM_'+infoplane_measure])[PLOT_LAYERS]
            ymvals = np.array(vals[epoch]['MI_YM_'+infoplane_measure])[PLOT_LAYERS]

            plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
            plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)
        
        plt.ylim([0, 3.5])
        plt.xlim([0, 14])
        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        plt.title(activation)
        
    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()

    if not os.path.isdir('plots'):
        os.mkdir('plots')
    plt.savefig('plots/' + args.activation+'_'+args.weights_config+'_infoplane_'+infoplane_measure,bbox_inches='tight')
    
# Plot SNR curves

plt.figure(figsize=(12,5))

gs = gridspec.GridSpec(len(measures), len(PLOT_LAYERS))
saved_data = {}
for actndx, activation in enumerate(measures.keys()):
    cur_dir = 'rawdata/' + args.activation+'_'+args.weights_config
    if not os.path.exists(cur_dir):
        continue
        
    epochs = []
    means = []
    stds = []
    wnorms = []
    trnloss = []
    tstloss = []
    for epochfile in sorted(os.listdir(cur_dir)):
        if not epochfile.startswith('epoch'):
            continue
            
        with open(cur_dir + "/"+epochfile, 'rb') as f:
            d = cPickle.load(f)
            
        epoch = d['epoch']
        epochs.append(epoch)
        wnorms.append(d['data']['weights_norm'])
        means.append(d['data']['gradmean'])
        stds.append(d['data']['gradstd'])
        trnloss.append(d['loss']['trn'])
        tstloss.append(d['loss']['tst'])

    wnorms, means, stds, trnloss, tstloss = map(np.array, [wnorms, means, stds, trnloss, tstloss])
    saved_data[activation] = {'wnorms':wnorms, 'means': means, 'stds': stds, 'trnloss': trnloss, 'tstloss':tstloss}

for lndx,layerid in enumerate(PLOT_LAYERS):
    plt.subplot(gs[actndx, lndx])
    plt.plot(epochs, means[:,layerid], 'b', label="Mean")
    plt.plot(epochs, stds[:,layerid], 'orange', label="Std")
    plt.plot(epochs, means[:,layerid]/stds[:,layerid], 'red', label="SNR")
    plt.plot(epochs, wnorms[:,layerid], 'g', label="||W||")

    plt.title('%s - Layer %d'%(activation, layerid))
    plt.xlabel('Epoch')
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_yscale("log", nonposy='clip')


plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.2))
plt.tight_layout()

plt.savefig('plots/' + args.activation+'_'+args.weights_config+'_snr', bbox_inches='tight')

GRID_PLOT_LAYERS = [0,1,2,3] # [1,2,3]
sns.set_style('whitegrid')
max_epoch = max( (max(vals.keys()) if len(vals) else 0) for vals in measures.values())
H_X = np.log2(10000)
for actndx, (activation, vals) in enumerate(measures.items()):
    fig = plt.figure(figsize=(12,11))
    gs = gridspec.GridSpec(4, len(GRID_PLOT_LAYERS))
    epochs = np.array(sorted(vals.keys()))
    if not len(epochs):
        continue
        
    plt.subplot(gs[0,0])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.plot(epochs,saved_data[activation]['trnloss']/np.log(2), label='Train')
    plt.plot(epochs,saved_data[activation]['tstloss']/np.log(2), label='Test')
    plt.ylabel('Cross entropy loss')
    plt.gca().set_xscale("log", nonposx='clip')
    
    plt.legend(loc='upper right', frameon=True)
        
    vals_binned = np.array([vals[epoch]['MI_XM_bin'] for epoch in epochs])
    vals_lower = np.array([vals[epoch]['MI_XM_lower'] for epoch in epochs])
    vals_upper = np.array([vals[epoch]['MI_XM_upper'] for epoch in epochs])
    for layerndx, layerid in enumerate(GRID_PLOT_LAYERS):
        plt.subplot(gs[1,layerndx])
        plt.plot(epochs, epochs*0 + H_X, 'k:', label=r'$H(X)$')
        plt.fill_between(epochs, vals_lower[:,layerid], vals_upper[:,layerid])
        plt.gca().set_xscale("log", nonposx='clip')
        plt.ylim([0, 1.1*H_X])
        plt.title('Layer %d Mutual Info (KDE)'%(layerid+1))
        plt.ylabel(r'$I(X;T)$')
        plt.xlabel('Epoch')
        if layerndx == len(GRID_PLOT_LAYERS)-1:
            plt.legend(loc='lower right', frameon=True)
        
        plt.subplot(gs[2,layerndx])
        plt.plot(epochs, epochs*0 + H_X, 'k:', label=r'$H(X)$')
        plt.plot(epochs, vals_binned[:,layerid])
        plt.gca().set_xscale("log", nonposx='clip')
        plt.ylim([0, 1.1*H_X])
        plt.ylabel(r'$I(X;T)$')
        plt.title('Layer %d Mutual Info (binned)'%(layerid+1))
        plt.xlabel('Epoch')
        if layerndx == len(GRID_PLOT_LAYERS)-1:
            plt.legend(loc='lower right', frameon=True)
        
        plt.subplot(gs[3,layerndx])
        plt.title('Layer %d SNR'%(layerid+1))
        plt.plot(epochs, saved_data[activation]['means'][:,layerid], 'b', label="Mean")
        plt.plot(epochs, saved_data[activation]['stds'][:,layerid], 'orange', label="Std")
        plt.plot(epochs, saved_data[activation]['means'][:,layerid]/saved_data[activation]['stds'][:,layerid], 'red', label="SNR")
        plt.plot(epochs, saved_data[activation]['wnorms'][:,layerid], 'g', label="||W||")

        plt.xlabel('Epoch')
        plt.gca().set_xscale("log", nonposx='clip')
        plt.gca().set_yscale("log", nonposy='clip')
    
    plt.tight_layout()
    plt.legend(loc='lower left', frameon=True)
    
    plt.savefig('plots/' + args.activation+'_'+args.weights_config+'_gridplot.png', bbox_inches='tight')

