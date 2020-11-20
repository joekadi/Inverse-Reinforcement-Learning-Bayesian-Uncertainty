function macro_reward = randomRgenerator()
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));
%rand('seed',seed);
macro_reward = (rand(1,1).^8)