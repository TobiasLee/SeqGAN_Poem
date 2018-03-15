from mydis import Discriminator
from mygen import Generator
from dataloader import Gen_Data_loader, Dis_dataloader
import random
import numpy as np
import tensorflow as tf
from myG_beta import G_beta


dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75

EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 10  # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 128
vocab_size = 6915 # max idx of word token = 6914

dis_emb_size = 64

TOTAL_BATCH = 200
positive_file = './train.txt'
negative_file = './generator_sample.txt'
eval_file = './eval_file.txt'
generated_num = 1000
sample_time = 16 # for G_beta to get reward
num_class = 2 # 0 : fake data 1 : real data

def main():
    # set random seed (may important to the result)
    np.random.seed(SEED)
    random.seed(SEED)

    # data loader
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)  # For testing

    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    D = Discriminator(SEQ_LENGTH, num_class, vocab_size, dis_emb_size, dis_filter_sizes, dis_num_filters, 0.2)
    G = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    # avoid occupy all the memory of the GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # change the train data to real poems  to be done
    gen_data_loader.create_batches(positive_file)

    log = open('./experiment-log.txt', 'w')
    #  pre-train generator
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, G, gen_data_loader)
        print("Epoch ", epoch, " loss: ", loss )
        # if epoch % 5 == 0:
        #     generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
        #     likelihood_data_loader.create_batches(eval_file)
        #     test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        #     print('pre-train epoch ', epoch, 'test_loss ', test_loss)
        #     buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
        #     log.write(buffer)
    print("Start pretraining the discriminator")
    for _ in range(50):
        generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    D.input_x: x_batch,
                    D.input_y: y_batch,
                    D.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(D.train_op, feed)

    g_beta = G_beta(G, update_rate=0.8)
    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')

    for total_batch in range(TOTAL_BATCH):
        # train generator once
        for it in range(1):
            samples = G.generate(sess)
            rewards = g_beta.get_reward(sess, samples, sample_time, D)
            feed = {G.x: samples, G.rewards: rewards}
            _ = sess.run(G.g_update, feed_dict=feed)
        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            # generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
            # likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\treward:\t' + str(rewards) + '\n'
            print('total_batch: ', total_batch, 'reward: ', rewards)
            log.write(buffer)

        # update G_beta with weight decay
        g_beta.update_params()

        # train the discriminator
        for it in range(10):
            generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for batch in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        D.input_x: x_batch,
                        D.input_y: y_batch,
                        D.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(D.train_op, feed_dict=feed)
# finnal generation
    print("Wrting final results to test file")
    test_file = "./final2.txt"
    generate_samples(sess, G, BATCH_SIZE, generated_num, test_file)
    print("Finished")

def generate_samples(sess, generator_model, batch_size, generated_num, output_file):

    generated_samples = []
    for i in range(generated_num // batch_size):
        one_batch = generator_model.generate(sess)
        generated_samples.extend(one_batch)
    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


# pre-train the Generator based on MLE method
def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


if __name__ == '__main__':
    main()