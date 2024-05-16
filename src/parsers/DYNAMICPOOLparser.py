#TODO implement for having arbitrary codes in the loop
# this will essentially be subclass of other parsers 
# but with collect_batch 
# and update_train_pool
# see loop in executor for methods to add

def update_pool_and_train(self, outputs):
    self.train = torch.concatenate((self.train, self.pool))
