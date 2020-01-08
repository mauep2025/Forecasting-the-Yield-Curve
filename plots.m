%%%Plot Yield Maturity

load ('yieldmxdata');
whos;


t=(2008:0.0833:2019.1667)'
t2=(2009:0.0833:2019.1667)'

s=[datamxyield.m_3(4) datamxyield.m_6(4) datamxyield.y_1(4) datamxyield.y_5(4) datamxyield.y_10(4) datamxyield.y_30(4); datamxyield.m_3(16) datamxyield.m_6(16) datamxyield.y_1(16) datamxyield.y_5(16) datamxyield.y_10(16) datamxyield.y_30(16); datamxyield.m_3(112) datamxyield.m_6(112) datamxyield.y_1(112) datamxyield.y_5(112) datamxyield.y_10(112) datamxyield.y_30(112) ; datamxyield.m_3(124) datamxyield.m_6(124) datamxyield.y_1(124) datamxyield.y_5(124) datamxyield.y_10(124) datamxyield.y_30(124); datamxyield.m_3(135) datamxyield.m_6(135) datamxyield.y_1(135) datamxyield.y_5(135) datamxyield.y_10(135) datamxyield.y_30(135)];

s1 = categorical ({'3-Month' '6-Month' '1-Year' '5-Year' '10-Year' '30-Year'});
%-----
figure;
plot(s1,s, 'LineWidth', 1.9) , title('Graph 2: Mexico Government Yield Curves', 'FontSize', 14), , legend({'April 2008','April 2009','April 2017','April 2018', 'March 2019' }, 'location','best'), legend('boxoff'), xlabel({'Source:Banco de Mexico'}), set(gca,'FontSize',12), set(gcf, 'color',  'w'), xlabel('Maturity');
ax=gca;
ax.XAxis.Categories;
ax.XAxis.Categories = {'3-Month', '6-Month', '1-Year', '5-Year', '10-Year', '30-Year'};
ylim([5.3 8.5]);



%%Plot Term Structure
figure;
plot(t, [datamxyield.m_3, datamxyield.m_6, datamxyield.y_1,  datamxyield.y_5, datamxyield.y_10, datamxyield.y_30],'LineWidth', 3), title('Graph 1: Mexico Government Bonds', 'FontSize', 14) , legend({'3-Month Bond','6-Month Bond','1-Year Bond','5-Year Bond','10-Year Bond','30-Year Bond'}, 'location','best'), legend('boxoff'), xlabel({'Source: Bloomberg'}), set(gca,'FontSize',12), set(gcf, 'color', 'w');

%%Plot Macroeconomic Variables 
figure;
subplot(3,2,1)
plot(t,datamxyield.rex, 'LineWidth', 1.5), title('Mexico: Real Exchange Rate', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,2)
plot(t,datamxyield.gdp, 'LineWidth', 1.5), title('Mexico: Global Indicator of Economic Activity', 'FontSize', 11), ylabel('Index'),  set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,3)
plot(t,datamxyield.tiie28, 'LineWidth', 1.5), title('Mexico: Interbank Equilibrum Interest Rate', 'FontSize', 11), ylabel('Percent'), set(gca,'FontSize',11), set(gcf, 'color',  'w');
subplot(3,2,4);
plot(t,datamxyield.oil_price, 'LineWidth', 1.5), title('US: Oil Price ', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11), set(gcf, 'color',  'w');
subplot(3,2,5);
plot(t,datamxyield.effr, 'LineWidth', 1.5), title('US: Effective Federal Funds Rate', 'FontSize', 11), ylabel('Percent'), set(gca,'FontSize',11), set(gcf, 'color',  'w');
xlabel('Source: INEGI, Banco de Mexico, BLS and FRB')
sgtitle('Graph 3: Macroeconomic Variables')