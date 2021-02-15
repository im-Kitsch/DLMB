import tensorflow as tf

event_summary = tf.compat.v1.train.summary_iterator(
    './runs/Feb04_10-49-08_u20_DCGAN_HAM10000/events.out.tfevents.1612432150.u20.4621.0'
)

for e in event_summary:
    for v in e.summary.value:
        if v.tag == 'img':
            with open(f'./runs/temp/{e.step}.png', 'wb') as f:
                f.write(v.image.encoded_image_string)
        print(v.tag)


