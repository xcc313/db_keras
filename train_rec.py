from absl import app, flags
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import datetime
from core.tools import build_cfg, generate_rec, DotDict
from core.model.r34vd_crnn import RecModel
from keras import callbacks
from keras import optimizers
from core.callbacks.bestkeepcallback import BestKeepCheckpoint
from core.callbacks.visual_callback import PredVisualize

flags.DEFINE_string('config', './configs/rec_r34_vd_ctc_captcha.yml', 'path to config file')
FLAGS = flags.FLAGS

def main(_argv):
    print(FLAGS.config)
    cfg = build_cfg(FLAGS.config)
    cfg.update({'mode': 'train'})
    cfg = DotDict.to_dot_dict(cfg)

    checkpoints_dir = f'checkpoints/{datetime.date.today()}'
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    model, inference_model, decode_model = RecModel(cfg)()
    model.summary()


    init_weight = cfg['train']['init_weight_path']

    train_generator = generate_rec(cfg['train'], cfg, is_training=True)
    val_generator = generate_rec(cfg['test'], cfg, is_training=False)

    try:
        if init_weight:
            print('Restoring weights from: %s ...' % init_weight)
            model.load_weights(init_weight, by_name=True, skip_mismatch=True)
            # convert to inference model
            # inference_model.save("captcha_model_0.918.h5")
        else:
            print('%s does not exist !' % init_weight)
            print('Training from scratch')

        checkpoint = callbacks.ModelCheckpoint(
            osp.join(checkpoints_dir, 'db_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'),
            verbose=1,
            save_weights_only=True,
        )

        bk = BestKeepCheckpoint(save_path=os.path.join(checkpoints_dir, "db_{epoch:02d}.h5"),
                                        eval_model=inference_model)

        visual = PredVisualize(inference_model, val_generator, cfg.char_ops)

        tb = callbacks.TensorBoard(
            log_dir="logs",
        )

        model.compile(optimizer=optimizers.Adam(lr=1.5e-3), loss={'CTCloss': lambda y_true, y_pred: y_pred})
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=125*4,
            initial_epoch=0,
            epochs=cfg.epochs,
            verbose=1,
            callbacks=[tb, bk, checkpoint],
            validation_data=val_generator,
            validation_steps=50
        )
    except Exception as e:
        print(e)

if __name__ == "__main__":
    app.run(main)
