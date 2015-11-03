# SmoothFlvSplitter
Split a flash movie into segments. Adobe Flash Player can smoothly play these segments without the POP-UP noise during segment switching.

It is a common industrial practice that VOD sites split the content into segments before streaming it to reduce bandwidth consumption.

For FLASH content, simple splitting the content will cause Pop-Up noises when the content is played by Adobe Flash Player embedded in the web page.  I have seen this issue on many video sites even some rather big ones.

This script helps to resolve this issue by carefully select several optimum split points in a very simple way.

I'd like to give special thanks to __Yusuke Shinyama__, who authored  the ['flv'](https://github.com/baijum/vnc2flv/blob/master/vnc2flv/flv.py) module of [__*'vnc2flv'*__](https://github.com/baijum/vnc2flv) project which comes ready handy in my project.

Please check the help for usages.  
