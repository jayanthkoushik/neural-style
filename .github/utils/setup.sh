cp index.md /www/
cp css/* /www/css/
cp js/* /www/js/
cp _includes/* /www/_includes/
cp -r imgs /www/imgs/
rm -r /www/fonts

cat js/_custom.ext.js >> /www/js/_custom.js
cat css/_custom.ext.scss >> /www/css/_custom.scss
