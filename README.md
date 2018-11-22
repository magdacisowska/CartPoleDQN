# CartPoleDQN
Deep Q Network for solving the CartPole from OpenAI Gym
* Neural net structure
![schemat](https://user-images.githubusercontent.com/28646893/48312011-b0e71200-e5a8-11e8-8c98-4ede5c35ea0e.JPG)
![siec](https://user-images.githubusercontent.com/28646893/48311985-41712280-e5a8-11e8-8c68-c8875f83ef5d.JPG)
* Three forces to apply
* Two directions
* Reward function: 
  - -20 for each loss (tilt larger than 12deg or moving out of the range of x (-4.2; 4.2))
  - for each step: R = 1 - |x_pos|
## Results
![ezgif-1-ddfb79880e2c](https://user-images.githubusercontent.com/28646893/48311927-6f099c00-e5a7-11e8-917a-6297dfac65fe.gif)
![learning](https://user-images.githubusercontent.com/28646893/48311992-58177980-e5a8-11e8-8e36-d8546f04229a.png)
![perf](https://user-images.githubusercontent.com/28646893/48311993-5d74c400-e5a8-11e8-8bbc-a1d1bf75de62.png)

## More evidence
![success](https://user-images.githubusercontent.com/28646893/48915025-7ec59200-ee7d-11e8-83a4-3d6df3b27647.png)
https://www.youtube.com/watch?v=J8FWwy-3TZ4&feature=youtu.be
