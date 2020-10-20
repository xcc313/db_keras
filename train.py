from absl import app, flags
import os.path as osp
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import datetime
from core.tools import build_cfg, generate
from core.model.r50vd_db import DetModel
from keras import callbacks
from keras import optimizers
from core.callbacks.bestkeepcallback import BestKeepCheckpoint


flags.DEFINE_string('config', './configs/det_r50_vd_db_colab.yml', 'path to config file')
FLAGS = flags.FLAGS


def main(_argv):
    print(FLAGS.config)
    cfg = build_cfg(FLAGS.config)
    cfg.update({'mode': 'train'})
    batch_size = cfg['train']['batch_size']

    checkpoints_dir = f'checkpoints/{datetime.date.today()}'
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    model_algorithm = cfg['det']['algorithm']
    if model_algorithm == 'DB':
        model, inference_model = DetModel(cfg)()
        model.summary()
    else:
        raise NotImplementedError('%s not support yet !' % model_algorithm)

    init_weight = cfg['train']['init_weight_path']

    train_generator = generate(cfg['train']['img_dir'], cfg['train']['label_path'], batch_size=batch_size)
    val_generator = generate(cfg['test']['img_dir'], cfg['test']['label_path'], batch_size=batch_size)

    try:
        if init_weight:
            print('Restoring weights from: %s ...' % init_weight)
            model.load_weights(init_weight, by_name=True, skip_mismatch=True)
            # convert to inference model
            # inference_model.save("db_inference_model.h5")
        else:
            print('%s does not exist !' % init_weight)
            print('Training from scratch')

        # lr_callback = [CosineAnnealingScheduler(learning_rate=1e-3,
        #                                         eta_min=1e-6,
        #                                         T_max=epochs * epoch_steps,
        #                                         verbose=1)]
        checkpoint = callbacks.ModelCheckpoint(
            osp.join(checkpoints_dir, 'db_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'),
            verbose=1,
        )

        bk = BestKeepCheckpoint(save_path=os.path.join(checkpoints_dir, "db_{epoch:02d}.h5"),
                                        eval_model=inference_model)
        tb = callbacks.TensorBoard(
            log_dir="logs",
        )

        model.compile(optimizer=optimizers.Adam(lr=1e-3), loss={'loss_all': lambda y_true, y_pred: y_pred})
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=125,
            initial_epoch=0,
            epochs=10,
            verbose=1,
            callbacks=[tb, bk, checkpoint],
            validation_data=val_generator,
            validation_steps=19
        )
    except Exception as e:
        print(e)




if __name__ == "__main__":
    app.run(main)
