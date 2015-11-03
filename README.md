It is a common industrial practice that VOD sites split the content into segments before streaming it to reduce bandwidth consumption.

For FLASH content, simple splitting the content will cause Pop-Up noises when the content is played by Adobe Flash Player embedded in the web page.  I have seen this issue on many video sites even some rather big ones.

This script helps to resolve this issue by carefully select several optimum split points in a very simple way.

I'd like to give special thanks to __Yusuke Shinyama__, who authored  the ['flv'](https://github.com/baijum/vnc2flv/blob/master/vnc2flv/flv.py) module of [__*'vnc2flv'*__](https://github.com/baijum/vnc2flv) project which comes rather handy in my project.

All the output segments have no meta data.  Please use ['yamdi'](http://yamdi.sourceforge.net/) to insert accurate meta data.
