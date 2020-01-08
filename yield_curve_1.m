%%
load ('yieldmxdata');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---------------ARIMA---------------------------------
%
AIC_spec=zeros(5,5);

porder=[0 1 2 3 4]';
qorder=[0 1 2 3 4]';
m_3 = datamxyield.m_3;
dm3= diff(m_3);

ixd=all(~ismissing(dm3),2);
idxPre=16:134;
TTT=ceil(.82*size(dm3,1));
idxEst=16:TTT;
idxF=(TTT+1):size(dm3,1);
fh=numel(idxF);

for i = 1:size(porder,1);
  p=porder(i);
    for j=1:size(qorder,1);
        q=qorder(j);
        model=arima(p,0,q);% p=AR order, d=order of difference, q=MA order
        [fit,VarCov,logL,info]=estimate(model,dm3(idxEst,:), 'Y0',dm3(idxPre,:));
        s2=fit.Variance;
        T=size(dm3,1);
        %LL = -(T/2)*log(det(e'*e./T))-(T*m/2)*(1+log(2*pi));
        %sbc(nmodel,1) = LL - (p+q)*log(T);
         AIC_spec(i,j)=T*log(s2)+  2*(p+q);               
    end            
end 



%%%%%%%%%% TO FORECAST Using AIC ARIMA minimising model %%%%%%%
% Select model with minimum AIC
[num idx] = min(AIC_spec(:))
[xaic yaic] = ind2sub(size(AIC_spec),idx);
% Estimate using preferred ARIMA model  
p=xaic-1;q=yaic-1;
model=arima(p,0,q);% p=AR order, d=order of difference, q=MA order
[fit,VarCov,logL,info]=estimate(model,dm3(idxEst,:), 'Y0',dm3(idxPre,:));
[res, ~, logL]=infer(fit,dm3(idxEst,:), 'Y0',dm3(idxPre,:));
res_arima_m3=res;
% Forecast h steps ahead 
h=fh;
[for_m_3,for_m_3mse,V]=forecast(fit,h,'Y0',dm3(idxEst,:));
aic_for=for_m_3;
mse_aic=for_m_3mse;
%Evaluation form ARIMA (Estimation Results)

RMSE_m3_aic_est=sqrt(mean(res_arima_m3.^2));

MAE1_aic_m3_est= mean(abs(res_arima_m3));

acf_Error_m3_aic_est = autocorr(res_arima_m3);

%Forecasting Evaluation form ARIMA (out-of*sample)

Error_m3_aic=dm3(idxF,1)-aic_for

RMSE_m3_aic=sqrt(mean(Error_m3_aic.^2));

MAE1_aic_m3= mean(abs(Error_m3_aic));

acf_Error_m3_aic = autocorr(Error_m3_aic);


%_-------------Ploting Forecast and Data ------

TTF = (2008.16667:0.083:2019.25)'
figure; 

    h1= plot(TTF(97:134),dm3(97:134), 'LineWidth', 1.6);
    hold on; 
    h2=plot(TTF(idxF), aic_for, 'LineWidth', 1 );
    h3=plot(TTF(idxF),aic_for+1.96*sqrt(mse_aic), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),aic_for-1.96*sqrt(mse_aic), 'k--', 'LineWidth', 1);
    title(['AIC ARIMA selected Forecast (3-Month Bond)']);
    ylim([-1 1.3])
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    set(gcf, 'color',  'w')
    legend('boxoff');
    hold off;

%Ploting the Forecast and the Results

idx_arima=all(~ismissing(m_3),2);
m_3_1=m_3(idx_arima,:);
[YPred_arima, YMSE_arima]=forecast(fit,24, 'Y0', dm3);
YFirst_arima=m_3_1(121:135);
EndPt_1= YFirst_arima(end,:);
EndPt_1(:,1:1)=log(EndPt_1);
YPred_arima=YPred_arima/100;
YPred_arima=[EndPt_1; YPred_arima];
YPred_arima(:,1:1)=cumsum(YPred_arima(:,1:1));
YPred_arima(:,1:1)=exp(YPred_arima(:,1:1));

TTFF2= (2019.25:0.083:2021.25)'
TTTF3= (2018.083:0.083:2019.25)'
figure
for j=1:1
    subplot(1,1,j)
    plot(TTFF2,YPred_arima(:,j),'--b')
    set(gcf, 'color',  'w')
    hold on 
    plot(TTTF3,YFirst_arima(:,j),'k')
    title('Mexico: 3-Month Bond')
    h=gca
    fill([TTFF2(1) h.XLim([2 2]) TTFF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    ylim([7.4 8.3])
    set(gcf, 'color',  'w')    
    hold off
end



%---------------------6-Month Bond-----------------------------
AIC_spec_m6=zeros(5,5);

m_6 = datamxyield.m_6
dm6= diff(m_6)
%%%%%% LOOP TO ESTIMATE RANGE OF ARMA MODELS %%%%%%%%%%%%%%


for i = 1:size(porder,1);
  p_m6=porder(i);
    for j=1:size(qorder,1);
        q_m6=qorder(j);
        model_m6=arima(p_m6,0,q_m6);% p=AR order, d=order of difference, q=MA order
        [fit_m6,VarCov_m6,logL_m6,info_m6]=estimate(model_m6,dm6(idxEst,:), 'Y0',dm6(idxPre,:));
        s2_m6=fit_m6.Variance;
        T_m6=size(dm6,1);
        %LL = -(T/2)*log(det(e'*e./T))-(T*m/2)*(1+log(2*pi));
        %sbc(nmodel,1) = LL - (p+q)*log(T);
       
        AIC_spec_m6(i,j)=T_m6*log(s2_m6)+  2*(p_m6+q_m6);               
    end            
end 



%%%%%%%%%% TO FORECAST Using AIC ARIMA minimising model %%%%%%%
% Select model with minimum AIC
[num_m6 idx_m6] = min(AIC_spec_m6(:))
[xaic_m6 yaic_m6] = ind2sub(size(AIC_spec_m6),idx_m6);
% Estimate using preferred ARIMA model  
p_m6=xaic_m6-1;q_m6=yaic_m6-1;
model_m6=arima(p_m6,0,q_m6);% p=AR order, d=order of difference, q=MA order
[fit_m6,VarCov_m6,logL_m6,info_m6]=estimate(model_m6,dm6(idxEst,:), 'Y0',dm6(idxPre,:));
[res, ~, logL]=infer(fit_m6,dm6(idxEst,:), 'Y0',dm6(idxPre,:));
res_arima_m6=res;
% Forecast 10 steps ahead 
h=fh;
[for_m_6,for_m_6mse,V_m6]=forecast(fit_m6,h,'Y0',dm6(idxEst,:));
aic_for_m6=for_m_6;
mse_aic_m6=for_m_6mse
%Evaluation form ARIMA (Estimation Results)

RMSE_m6_aic_est=sqrt(mean(res_arima_m6.^2));

MAE1_aic_m6_est= mean(abs(res_arima_m6));

acf_Error_m6_aic_est = autocorr(res_arima_m6);

%Forecasting Evaluation form ARIMA 

Error_m6_aic=dm6(idxF,1)-aic_for_m6

RMSE_m6_aic=sqrt(mean(Error_m6_aic.^2));

MAE1_aic_m6= mean(abs(Error_m6_aic));

acf_Error_m6_aic = autocorr(Error_m6_aic);




%_-------------Ploting Forecast and Data ------

TTF = (2008.16667:0.083:2019.25)'
figure; 

    h1= plot(TTF(97:134),dm6(97:134), 'LineWidth', 1.6);
    hold on; 
    h2=plot(TTF(idxF), aic_for_m6, 'LineWidth', 1);
    h3=plot(TTF(idxF),aic_for_m6+1.96*sqrt(mse_aic_m6), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),aic_for_m6-1.96*sqrt(mse_aic_m6), 'k--', 'LineWidth', 1);
    title(['AIC ARIMA selected Forecast (6-Month Bond)']);
    ylim([-1 1])
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    set(gcf, 'color',  'w');
    legend('boxoff');
    hold off;

%Ploting the Forecast and the Results

idx_arima_m6=all(~ismissing(m_6),2);
m_6_1=m_6(idx_arima_m6,:);
[YPred_arima_m6, YMSE_arima_m6]=forecast(fit_m6,24, 'Y0', dm6);
YFirst_arima_m6=m_6_1(121:135);
EndPt_1_m6= YFirst_arima_m6(end,:);
EndPt_1_m6(:,1:1)=log(EndPt_1_m6);
YPred_arima_m6=YPred_arima_m6/100;
YPred_arima_m6=[EndPt_1_m6; YPred_arima_m6];
YPred_arima_m6(:,1:1)=cumsum(YPred_arima_m6(:,1:1));
YPred_arima_m6(:,1:1)=exp(YPred_arima_m6(:,1:1));

TTFF2= (2019.25:0.083:2021.25)'
TTTF3= (2018.083:0.083:2019.25)'
figure
for j=1:1
    subplot(1,1,j)
    plot(TTFF2,YPred_arima_m6(:,j),'--b')
    set(gcf, 'color',  'w')
    hold on 
    plot(TTTF3,YFirst_arima_m6(:,j),'k')
    set(gcf, 'color',  'w');
    title('Mexico: 6-Month Bond ')
    h=gca
    fill([TTFF2(1) h.XLim([2 2]) TTFF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    ylim([7.6 8.6])
    hold off
end

%----------------------1-Year-----------------------
AIC_spec_y1=zeros(5,5);

y_1 = datamxyield.y_1
dy1= diff(y_1);


for i = 1:size(porder,1);
  p_y1=porder(i);
    for j=1:size(qorder,1);
        q_y1=qorder(j);
        model_y1=arima(p_y1,0,q_y1);% p=AR order, d=order of difference, q=MA order
        [fit_y1,VarCov_y1,logL_y1,info_y1]=estimate(model_y1,dy1(idxEst,:), 'Y0',dy1(idxPre,:));
        s2_y1=fit_y1.Variance;
        T_y1=size(dy1,1);
             
        AIC_spec_y1(i,j)=T_y1*log(s2_y1)+  2*(p_y1+q_y1);               
    end            
end 



%%%%%%%%%% TO FORECAST Using AIC ARIMA minimising model %%%%%%%
% Select model with minimum AIC
[num_y1, idx_y1] = min(AIC_spec_y1(:))
[xaic_y1 yaic_y1] = ind2sub(size(AIC_spec_y1),idx_y1);
% Estimate using preferred ARIMA model  
p_y1=xaic_y1-1;q_y1=yaic_y1-1;
model_y1=arima(p_y1,0,q_y1);% p=AR order, d=order of difference, q=MA order
[fit_y1,VarCov_y1,logL_y1,info_y1]=estimate(model_y1,dy1(idxEst,:), 'Y0',dy1(idxPre,:));
[res, ~, logL]=infer(fit_y1,dy1(idxEst,:), 'Y0',dy1(idxPre,:));
res_arima_y1=res;
% Forecast 10 steps ahead 
h=fh;
[for_y1,for_y1_mse,V_y1]=forecast(fit_y1,h,'Y0',dy1(idxEst,:));
aic_for_y1=for_y1;
mse_aic_y1=for_y1_mse
%Evaluation form ARIMA (Estimation Results)

RMSE_y1_aic_est=sqrt(mean(res_arima_y1.^2));

MAE1_aic_y1_est= mean(abs(res_arima_y1));

acf_Error_y1_aic_est = autocorr(res_arima_y1);

%Forecasting Evaluation form ARIMA 

Error_y1_aic=dy1(idxF,1)-aic_for_y1

RMSE_y1_aic=sqrt(mean(Error_y1_aic.^2));

MAE1_aic_y1= mean(abs(Error_y1_aic));

acf_Error_y1_aic = autocorr(Error_y1_aic);





%_-------------Ploting Forecast and Data ------

TTF_y1 = (2008.16667:0.083:2019.25)'
figure; 
    h1= plot(TTF_y1(97:134),dy1(97:134), 'LineWidth', 1.6);
    hold on; 
    h2=plot(TTF_y1(idxF), aic_for_y1, 'LineWidth', 1);
    h3=plot(TTF_y1(idxF),aic_for_y1+1.96*sqrt(mse_aic_y1), 'k--', 'LineWidth', 1);
    plot(TTF_y1(idxF),aic_for_y1-1.96*sqrt(mse_aic_y1), 'k--', 'LineWidth', 1);
    title(['AIC ARIMA selected Forecast (1-Year Bond)']);
    ylim([-.5 1])
    h=gca;
    fill([TTF_y1(idxF(1)) h.XLim([2 2]) TTF_y1(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    set(gcf, 'color',  'w');
    legend('boxoff');
    hold off;




%Ploting the Forecast and the Results

idx_arima_y1=all(~ismissing(y_1),2);
y1_1=y_1(idx_arima_y1,:);
[YPred_arima_y1, YMSE_arima_y1]=forecast(fit_y1,24, 'Y0', dy1);
YFirst_arima_y1=y1_1(121:135);
EndPt_1_y1= YFirst_arima_y1(end,:);
EndPt_1_y1(:,1:1)=log(EndPt_1_y1);
YPred_arima_y1=YPred_arima_y1/100;
YPred_arima_y1=[EndPt_1_y1; YPred_arima_y1];
YPred_arima_y1(:,1:1)=cumsum(YPred_arima_y1(:,1:1));
YPred_arima_y1(:,1:1)=exp(YPred_arima_y1(:,1:1));

TTFF2= (2019.25:0.083:2021.25)'
TTTF3= (2018.083:0.083:2019.25)'
figure
for j=1:1
    subplot(1,1,j)
    plot(TTFF2,YPred_arima_y1(:,j),'--b')
    set(gcf, 'color',  'w')
    hold on 
    plot(TTTF3,YFirst_arima_y1(:,j),'k')
    title('Mexico: 1-Year Bond ')
    h=gca
    fill([TTFF2(1) h.XLim([2 2]) TTFF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    ylim([7.6 8.7])
    set(gcf, 'color',  'w');
    hold off
end


%----------------------5-Year-----------------------
AIC_spec_y5=zeros(5,5);
BIC_spec_y5=zeros(5,5);
y_5 = datamxyield.y_5
dy5= diff(y_5);

for i = 1:size(porder,1);
  p_y5=porder(i);
    for j=1:size(qorder,1);
        q_y5=qorder(j);
        model_y5=arima(p_y5,0,q_y5);% p=AR order, d=order of difference, q=MA order
        [fit_y5,VarCov_y5,logL_y5,info_y5]=estimate(model_y5,dy5(idxEst,:), 'Y0',dy5(idxPre,:));
        s2_y5=fit_y5.Variance;
        T_y5=size(dy5,1);
    
       
        AIC_spec_y5(i,j)=T_y5*log(s2_y5)+  2*(p_y5+q_y5);               
    end            
end 



%%%%%%%%%% TO FORECAST Using AIC ARIMA minimising model %%%%%%%
% Select model with minimum AIC
[num_y5 idx_y5] = min(AIC_spec_y5(:))
[xaic_y5 yaic_y5] = ind2sub(size(AIC_spec_y5),idx_y5);
% Estimate using preferred ARIMA model  
p_y5=xaic_y5-1;q_y5=yaic_y5-1;
model_y5=arima(p_y5,0,q_y5);% p=AR order, d=order of difference, q=MA order
[fit_y5,VarCov_y5,logL_y5,info_y5]=estimate(model_y5,dy5(idxEst,:), 'Y0',dy5(idxPre,:));
[res, ~, logL]=infer(fit_y5,dy5(idxEst,:), 'Y0',dy5(idxPre,:));
res_arima_y5=res;
% Forecast 10 steps ahead 
h=fh;
[for_y5,for_y5_mse,V_y5]=forecast(fit_y5,h,'Y0',dy5(idxEst,:));
aic_for_y5=for_y5;
mse_aic_y5=for_y5_mse;
%Evaluation form ARIMA (Estimation Results)

RMSE_y5_aic_est=sqrt(mean(res_arima_y5.^2));

MAE1_aic_y5_est= mean(abs(res_arima_y5));

acf_Error_y5_aic_est = autocorr(res_arima_y5);
%Forecasting Evaluation form ARIMA 

Error_aic_y5=dy5(idxF,1)-aic_for_y5;

RMSE_aic_y5=sqrt(mean(Error_aic_y5.^2));

MAE1_aic_y5= mean(abs(Error_aic_y5));

acf_Error_y5_aic = autocorr(Error_aic_y5);


%_-------------Ploting Forecast and Data ------

TTF = (2008.16667:0.083:2019.25)';
figure; 

%subplot(2,1,2);
    h1= plot(TTF(97:134),dy5(97:134), 'LineWidth', 1.6);
    hold on; 
    h2=plot(TTF(idxF), aic_for_y5, 'LineWidth', 1);
    h3=plot(TTF(idxF),aic_for_y5+1.96*sqrt(mse_aic_y5), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),aic_for_y5-1.96*sqrt(mse_aic_y5), 'k--', 'LineWidth', 1);
    title(['AIC ARIMA selected Forecast (5-Year Bond)']);
    ylim([-1 1])
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    set(gcf, 'color',  'w');
    legend('boxoff');
    hold off;


%Ploting the Forecast and the Results

idx_arima_y5=all(~ismissing(y_5),2);
y_5_1=y_5(idx_arima_y5,:);
[YPred_arima_y5, YMSE_arima_y5]=forecast(fit_y5,24, 'Y0', dy5);
YFirst_arima_y5=y_5_1(121:135);
EndPt_1_y5= YFirst_arima_y5(end,:);
EndPt_1_y5(:,1:1)=log(EndPt_1_y5);
YPred_arima_y5=YPred_arima_y5/100;
YPred_arima_y5=[EndPt_1_y5; YPred_arima_y5];
YPred_arima_y5(:,1:1)=cumsum(YPred_arima_y5(:,1:1));
YPred_arima_y5(:,1:1)=exp(YPred_arima_y5(:,1:1));

TTFF2= (2019.25:0.083:2021.25)'
TTTF3= (2018.083:0.083:2019.25)'
figure
for j=1:1
    subplot(1,1,j)
    plot(TTFF2,YPred_arima_y5(:,j),'--b')
    set(gcf, 'color',  'w')
    hold on 
    plot(TTTF3,YFirst_arima_y5(:,j),'k')
    title('Mexico: 5-Year Bond ')
    h=gca
    fill([TTFF2(1) h.XLim([2 2]) TTFF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    set(gcf, 'color',  'w');
    hold off
end



%----------------------10-Year-----------------------

AIC_spec_y10=zeros(5,5);
BIC_spec_y10=zeros(5,5);
y_10 = datamxyield.y_10
dy10= diff(y_10);

for i = 1:size(porder,1);
  p_y10=porder(i);
    for j=1:size(qorder,1);
        q_y10=qorder(j);
        model_y10=arima(p_y10,0,q_y10);% p=AR order, d=order of difference, q=MA order
        [fit_y10,VarCov_y10,logL_y10,info_y10]=estimate(model_y10,dy10(idxEst,:), 'Y0',dy10(idxPre,:));
        s2_y10=fit_y10.Variance;
        T_y10=size(dy10,1);
        %LL = -(T/2)*log(det(e'*e./T))-(T*m/2)*(1+log(2*pi));
        %sbc(nmodel,1) = LL - (p+q)*log(T);
        
        AIC_spec_y10(i,j)=T_y10*log(s2_y10)+  2*(p_y10+q_y10);               
    end            
end 



%%%%%%%%%% TO FORECAST Using AIC ARIMA minimising model %%%%%%%
% Select model with minimum AIC
[num_y10 idx_y10] = min(AIC_spec_y10(:))
[xaic_y10 yaic_y10] = ind2sub(size(AIC_spec_y10),idx_y10);
% Estimate using preferred ARIMA model  
p_y10=xaic_y10-1;q_y10=yaic_y10-1;
model_y10=arima(p_y10,0,q_y10);% p=AR order, d=order of difference, q=MA order
[fit_y10,VarCov_y10,logL_y10,info_y10]=estimate(model_y10,dy10(idxEst,:), 'Y0',dy10(idxPre,:));
[res, ~, logL]=infer(fit_y10,dy10(idxEst,:), 'Y0',dy10(idxPre,:));
res_arima_y10=res; 
% Forecast 10 steps ahead 
h=fh;
[for_y10,for_y10_mse,V_y10]=forecast(fit_y10,h,'Y0',dy10(idxEst,:));
aic_for_y10=for_y10;
mse_aic_y10=for_y10_mse;
%Evaluation form ARIMA (Estimation Results)

RMSE_y10_aic_est=sqrt(mean(res_arima_y10.^2));

MAE1_aic_y10_est= mean(abs(res_arima_y10));

acf_Error_y10_aic_est = autocorr(res_arima_y10);
%Forecasting Evaluation form ARIMA 

Error_aic_y10=dy10(idxF,1)-aic_for_y10;

RMSE_aic_y10=sqrt(mean(Error_aic_y10.^2));

MAE1_aic_y10= mean(abs(Error_aic_y10));

acf_Error_y10_aic = autocorr(Error_aic_y10);
	




%_-------------Ploting Forecast and Data ------

TTF = (2008.16667:0.083:2019.25)'
figure; 
    h1= plot(TTF(97:134),dy10(97:134), 'LineWidth', 1.6);
    hold on; 
    h2=plot(TTF(idxF), aic_for_y10, 'LineWidth', 1);
    h3=plot(TTF(idxF),aic_for_y10+1.96*sqrt(mse_aic_y10), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),aic_for_y10-1.96*sqrt(mse_aic_y10), 'k--', 'LineWidth', 1);
    title(['AIC ARIMA selected Forecast (10-Year Bond)']);
    ylim([-1 1])
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    set(gcf, 'color',  'w');
    legend('boxoff');
    hold off;

%Ploting the Forecast and the Results

idx_arima_y10=all(~ismissing(y_10),2);
y_10_1=y_10(idx_arima_y10,:);
[YPred_arima_y10, YMSE_arima_y10]=forecast(fit_y10,24, 'Y0', dy10);
YFirst_arima_y10=y_10_1(121:135);
EndPt_1_y10= YFirst_arima_y10(end,:);
EndPt_1_y10(:,1:1)=log(EndPt_1_y10);
YPred_arima_y10=YPred_arima_y10/100;
YPred_arima_y10=[EndPt_1_y10; YPred_arima_y10];
YPred_arima_y10(:,1:1)=cumsum(YPred_arima_y10(:,1:1));
YPred_arima_y10(:,1:1)=exp(YPred_arima_y10(:,1:1));

TTFF2= (2019.25:0.083:2021.25)'
TTTF3= (2018.083:0.083:2019.25)'
figure
for j=1:1
    subplot(1,1,j)
    plot(TTFF2,YPred_arima_y10(:,j),'--b')
    set(gcf, 'color',  'w')
    hold on 
    plot(TTTF3,YFirst_arima_y10(:,j),'k')
    title('Mexico: 10-Year Bond ')
    h=gca
    fill([TTFF2(1) h.XLim([2 2]) TTFF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    set(gcf, 'color',  'w');
    hold off
end

%----------------------30-Year-----------------------


AIC_spec_y30=zeros(5,5);
BIC_spec_y30=zeros(5,5);
y_30 = datamxyield.y_30
dy30= diff(y_30)
%%%%%% LOOP TO ESTIMATE RANGE OF ARMA MODELS %%%%%%%%%%%%%%

for i = 1:size(porder,1);
  p_y30=porder(i);
    for j=1:size(qorder,1);
        q_y30=qorder(j);
        model_y30=arima(p_y30,0,q_y30);% p=AR order, d=order of difference, q=MA order
        [fit_y30,VarCov_y30,logL_y30,info_y30]=estimate(model_y30,dy30(idxEst,:), 'Y0',dy30(idxPre,:));
        s2_y30=fit_y30.Variance;
        T_y30=size(dy30,1);
             
        AIC_spec_y30(i,j)=T_y30*log(s2_y30)+  2*(p_y30+q_y30);               
    end            
end 



%%%%%%%%%% TO FORECAST Using AIC ARIMA minimising model %%%%%%%
% Select model with minimum AIC
[num_y30 idx_y30] = min(AIC_spec_y30(:))
[xaic_y30 yaic_y30] = ind2sub(size(AIC_spec_y30),idx_y30);
% Estimate using preferred ARIMA model  
p_y30=xaic_y30-1;q_y30=yaic_y30-1;
model_y30=arima(p_y30,0,q_y30);% p=AR order, d=order of difference, q=MA order
[fit_y30,VarCov_y30,logL_y30,info_y30]=estimate(model_y30,dy30(idxEst,:), 'Y0',dy30(idxPre,:));
[res, ~, logL]=infer(fit_y30,dy30(idxEst,:), 'Y0',dy30(idxPre,:));
res_arima_y30=res;
% Forecast 10 steps ahead 
h=fh;
[for_y30,for_y30_mse,V_y30]=forecast(fit_y30,h,'Y0',dy30(idxEst,:));
aic_for_y30=for_y30;
mse_aic_y30=for_y30_mse;
%Evaluation form ARIMA (Estimation Results)

RMSE_y30_aic_est=sqrt(mean(res_arima_y30.^2));

MAE1_aic_y30_est= mean(abs(res_arima_y30));

acf_Error_y30_aic_est = autocorr(res_arima_y30);
%Forecasting Evaluation form ARIMA 

Error_aic_y30=dy30(idxF,1)-aic_for_y30;

RMSE_aic_y30=sqrt(mean(Error_aic_y30.^2));


MAE1_aic_y30= mean(abs(Error_aic_y30));

acf_Error_y30_aic = autocorr(Error_aic_y30);


%_-------------Ploting Forecast and Data ------

TTF = (2008.16667:0.083:2019.25)'
figure; 
 
    h1= plot(TTF(97:134),dy30(97:134), 'LineWidth', 1.6);
    hold on; 
    h2=plot(TTF(idxF), aic_for_y30, 'LineWidth', 1);
    h3=plot(TTF(idxF),aic_for_y30+1.96*sqrt(mse_aic_y30), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),aic_for_y30-1.96*sqrt(mse_aic_y30), 'k--', 'LineWidth', 1);
    title(['AIC ARIMA selected Forecast (30-Year Bond)']);
    ylim([-1.5 1.2])
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    legend('boxoff');
    set(gcf, 'color',  'w');
    hold off;


%Ploting the Forecast and the Results

idx_arima_y30=all(~ismissing(y_30),2);
y_30_1=y_30(idx_arima_y30,:);
[YPred_arima_y30, YMSE_arima_y30]=forecast(fit_y30,24, 'Y0', dy30);
YFirst_arima_y30=y_30_1(121:135);
EndPt_1_y30= YFirst_arima_y30(end,:);
EndPt_1_y30(:,1:1)=log(EndPt_1_y30);
YPred_arima_y30=YPred_arima_y30/100;
YPred_arima_y30=[EndPt_1_y30; YPred_arima_y30];
YPred_arima_y30(:,1:1)=cumsum(YPred_arima_y30(:,1:1));
YPred_arima_y30(:,1:1)=exp(YPred_arima_y30(:,1:1));

TTFF2= (2019.25:0.083:2021.25)'
TTTF3= (2018.083:0.083:2019.25)'
figure
for j=1:1
    subplot(1,1,j)
    plot(TTFF2,YPred_arima_y30(:,j),'--b')
    set(gcf, 'color',  'w')
    hold on 
    plot(TTTF3,YFirst_arima_y30(:,j),'k')
    title('Mexico: 30-Year Bond')
    h=gca
    fill([TTFF2(1) h.XLim([2 2]) TTFF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    hold off
end


%---------------------------------
%--------------------------------- VAR

load ('yieldmxdata');
dm_3=diff(datamxyield.m_3);
dm_1=diff(datamxyield.m_1);
dm_6=diff(datamxyield.m_6);
dy1=diff(datamxyield.y_1);
dy3=diff(datamxyield.y_3);
dy5=diff(datamxyield.y_5);
dy7=diff(datamxyield.y_7);
dy10=diff(datamxyield.y_10);
dy20=diff(datamxyield.y_20);
dy30=diff(datamxyield.y_30);
rexg=diff(datamxyield.rex);
gdpg=diff(datamxyield.gdp);
inflg=diff(datamxyield.infl);
indusag=diff(datamxyield.indusa);
effrg = diff(datamxyield.effr);
influsg =diff(datamxyield.infl_usa);
oilg=diff(datamxyield.oil_price);
tiie28g=diff(datamxyield.tiie28);

m_3=datamxyield.m_3;
m_1=datamxyield.m_1;
m_6=datamxyield.m_6;
y1=datamxyield.y_1;
y3=datamxyield.y_3;
y5=datamxyield.y_5;
y7=datamxyield.y_7;
y10=datamxyield.y_10;
y20=datamxyield.y_20;
y30=datamxyield.y_30;

rex=datamxyield.rex;
gdp=datamxyield.gdp;
infl=datamxyield.infl;
indusa=datamxyield.indusa;
effr = datamxyield.effr;
influs =datamxyield.infl_usa;
oil=datamxyield.oil_price;
tiie28=datamxyield.tiie28;



A1 = table(dm_3,rexg,gdpg,tiie28g,oilg,effrg);
A2 = table(m_3,rex,gdp,tiie28,oil,effr);
ixd=all(~ismissing(A1),2);
ixd2=all(~ismissing(A2),2);
A1=A1(ixd,:);
A2=A2(ixd2,:);
idxPre=16:134;
TTT=ceil(.82*size(A1,1));
TTTT_1=ceil(.82*size(A2,1));
idxEst=16:TTT;
idxEst_1=16:TTTT_1;
idxF=(TTT+1):size(A1,1);
idxF_2=(TTT+1):size(A2,1);
fh=numel(idxF);
fh_2=numel(idxF_2);

 %%% Select and Fit the Model
 

numseries=6;
seriesnam={'3-Month Bond ',  'Mexico: Real Exchange Rate', 'Mexico: Global Economic Indicator', 'Mexico: Central Bank Funds Rate','US: Oil Price', 'US: Effective Federal Funds Rate'  };
VAR1f=varm(numseries,1);
VAR1f.SeriesNames=seriesnam;
VAR2f=varm(numseries,2);
VAR2f.SeriesNames=seriesnam;
VAR3f=varm(numseries,3);
VAR3f.SeriesNames=seriesnam;

[EstMdl1,EstSE1,logL1,E1]= estimate(VAR1f,A1{idxEst,:}, 'Y0',A1{idxPre,:});
[EstMdl2,EstSE2,logL2,E2]= estimate(VAR2f,A1{idxEst,:}, 'Y0',A1{idxPre,:});
[EstMdl3,EstSE3,logL3,E3]= estimate(VAR3f,A1{idxEst,:}, 'Y0',A1{idxPre,:});

%%Check Model Adequacy
EstMdl1.Description
EstMdl2.Description
EstMdl3.Description

results1=summarize(EstMdl1);
np1=results1.NumEstimatedParameters;
results2=summarize(EstMdl2);
np2=results2.NumEstimatedParameters;
results3=summarize(EstMdl3);
np3=results3.NumEstimatedParameters;
%%Likelihood Ratio test of model specification 
[h,pValue,stat,cValue]=lratiotest(logL2,logL1,np2- np1)
[h,pValue,stat,cValue]=lratiotest(logL3,logL1,np3- np1)

%%%Akaike Information Criterion (AIC)

AIC=aicbic([logL1 logL2 logL3], [np1 np2 np3]);

%%Comparing the predictions 

[FY1,FYCov1]=forecast(EstMdl1,fh,A1{idxEst,:});
[FY2,FYCov2]=forecast(EstMdl2,fh,A1{idxEst,:});
[FY3,FYCov3]=forecast(EstMdl3,fh,A1{idxEst,:});

%Estimating the 95% forecast interval for the best fitting model
extractMSE=@(x)diag(x)';
MSE=cellfun(extractMSE, FYCov2, 'UniformOutput', false);
SE=sqrt(cell2mat(MSE));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% Change
YFI= zeros(fh,EstMdl2.NumSeries,2);
YFI(:,:,1) = FY2 - 1.96*SE; %Lower Band
YFI(:,:,2) = FY2 + 1.96*SE; %Upper Band


%plotting the results
TTF = (2008.16667:0.083:2019.25)'

for j=1:1
    subplot(1,1,j);
    h1= plot(TTF((end-30):end),A1{(end-30):end,j});
    hold on;
    h2=plot(TTF(idxF),FY2(:,j));
    h3=plot(TTF(idxF),YFI(:,j,1),'k--');
    plot(TTF(idxF),YFI(:,j,2),'k--');
    title('Model Adequacy: VAR best fitting model (3-Month Bond)');
   
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'southwest')
    legend('boxoff');
    set(gcf, 'color',  'w');
    ylim([-.4 1]);
    hold off;
end    
figure; 

%Evaluation form ARIMA (Estimation Results)

RMSE_VAR1_est_m3=sqrt(mean(E1(:,1).^2));
RMSE_VAR2_est_m3=sqrt(mean(E2(:,1).^2));
RMSE_VAR3_est_m3=sqrt(mean(E3(:,1).^2));

MAE1_VAR1_est_m3= mean(abs(E1(:,1)));
MAE2_VAR2_est_m3= mean(abs(E2(:,1)));
MAE3_VAR3_est_m3= mean(abs(E3(:,1)));

acf_Error1_VAR1_est_m3 = autocorr(E1(:,1));
acf_Error2_VAR2_est_m3 = autocorr(E2(:,1));
acf_Error3_VAR3_est_m3 = autocorr(E3(:,1));

%%%%%Sum-of-squares error between the predictors and the observed figures

Error1=A1{idxF,:}-FY1;
Error2=A1{idxF,:}-FY2;
Error3=A1{idxF,:}-FY3;

RMSEVAR1=sqrt(mean(Error1(:,1).^2))
RMSEVAR2=sqrt(mean(Error2(:,1).^2))
RMSEVAR3=sqrt(mean(Error3(:,1).^2))

MAE1= mean(abs(Error1(:,1)))
MAE2= mean(abs(Error2(:,1)))
MAE3= mean(abs(Error3(:,1)))

acf_Error1_m3 = autocorr(Error1(:,1));
acf_Error2_m3 = autocorr(Error2(:,1));
acf_Error3_m3 = autocorr(Error3(:,1));


%According with the RMSE test the best model for Forecast is the VAR2

%For the forecast 

SSerror1=Error1(:)'*Error1(:);
SSerror2=Error2(:)'*Error2(:);
SSerror3=Error1(:)'*Error3(:);

summarize(EstMdl1);

%%Forecast Observations
[YPred, YCov]=forecast(EstMdl2,24,A1{10:134,:});

YFirst = A2(ixd2, {'m_3' 'rex' 'gdp' 'tiie28' 'oil' 'effr'});
YFirst = A2(121:135,:);

EndPt = YFirst{end,:};
EndPt(:,1:6) = log(EndPt(:,1:6));
YPred(:,1:6)=YPred(:,1:6)/100;
YPred=[EndPt; YPred];
YPred(:,1:6)=cumsum(YPred(:,1:6));
YPred(:,1:6)=exp(YPred(:,1:6));
TTF2= (2019.25:0.083:2021.25)'
TTT3= (2008.083:0.083:2019.25)'
%%%%%
%--------------
 
    plot(TTF2,YPred(:,1),'--b');
    hold on 
    plot(TTT3(121:135),YFirst{:,1},'k');
    title('Forecast 24-months ahead (3-Month Bond)');
    h=gca;
    fill([TTF2(1) h.XLim([2 2]) TTF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    set(gcf, 'color',  'w');
    hold off
    ylim([7.4 8.3]);



%----------------6-Month Boond

A3 = table(dm_6,rexg,gdpg,tiie28g,oilg,effrg); 
A4 = table(m_6,rex,gdp,tiie28,oil,effr);
ixd=all(~ismissing(A3),2);
ixd2=all(~ismissing(A4),2);
A3=A3(ixd,:);
A4=A4(ixd2,:);
idxPre=16:134;
TTT=ceil(.82*size(A3,1));
TTTT_1=ceil(.82*size(A4,1));
idxEst=16:TTT;
idxEst_1=16:TTTT_1;
idxF=(TTT+1):size(A3,1);
idxF_2=(TTT+1):size(A4,1);
fh=numel(idxF);
fh_2=numel(idxF_2);

 %%% Select and Fit the Model
 

numseries=6;
seriesnam_m6={'6-Month Bond ','Mexico: Real Exchange Rate', 'Mexico: Global Economic Indicator', 'Mexico: Central Bank Funds Rate','US: Oil Price', 'US: Effective Federal Funds Rate'  };
VAR1f_m6=varm(numseries,1);
VAR1f_m6.SeriesNames=seriesnam_m6;
VAR2f_m6=varm(numseries,2);
VAR2f_m6.SeriesNames=seriesnam_m6;
VAR3f_m6=varm(numseries,3);
VAR3f_m6.SeriesNames=seriesnam_m6;

[EstMdl1_m6,EstSE1_m6,logL1_m6,E1_m6]= estimate(VAR1f_m6,A3{idxEst,:}, 'Y0',A3{idxPre,:});
[EstMdl2_m6,EstSE2_m6,logL2_m6,E2_m6]= estimate(VAR2f_m6,A3{idxEst,:}, 'Y0',A3{idxPre,:});
[EstMdl3_m6,EstSE3_m6,logL3_m6,E3_m6]= estimate(VAR3f_m6,A3{idxEst,:}, 'Y0',A3{idxPre,:});

%%Check Model Adequacy
EstMdl1_m6.Description
EstMdl2_m6.Description
EstMdl3_m6.Description

results1_m6=summarize(EstMdl1_m6);
np1_m6=results1_m6.NumEstimatedParameters;
results2_m6=summarize(EstMdl2_m6);
np2_m6=results2_m6.NumEstimatedParameters;
results3_m6=summarize(EstMdl3_m6);
np3_m6=results3_m6.NumEstimatedParameters;
%%Likelihood Ratio test of model specification 
[h_m6,pValue_m6,stat_m6,cValue_m6]=lratiotest(logL2_m6,logL1_m6,np2_m6- np1_m6)
[h1_m6,pValue1_m6,stat1_m6,cValue1_m6]=lratiotest(logL3_m6,logL1_m6,np3_m6- np1_m6)
%%Missing a matrix which store the information for the LM test

%%%Akaike Information Criterion (AIC)

AIC_m6=aicbic([logL1_m6 logL2_m6 logL3_m6], [np1_m6 np2_m6 np3_m6]);

%For the 3-Month the best model id the VAR1

%%Comparing the predictions 

[FY1_m6,FYCov1_m6]=forecast(EstMdl1_m6,fh,A3{idxEst,:});
[FY2_m6,FYCov2_m6]=forecast(EstMdl2_m6,fh,A3{idxEst,:});
[FY3_m6,FYCov3_m6]=forecast(EstMdl3_m6,fh,A3{idxEst,:});

%Estimating the 95% forecast interval for the best fitting model

extractMSE=@(x)diag(x)';
MSE_m6=cellfun(extractMSE, FYCov1_m6, 'UniformOutput', false);
SE_m6=sqrt(cell2mat(MSE_m6));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% Change
YFI_m6= zeros(fh,EstMdl2_m6.NumSeries,2);
YFI_m6(:,:,1) = FY1_m6 - 2*SE_m6; %Lower Band
YFI_m6(:,:,2) = FY1_m6 + 2*SE_m6; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


%%plotting the results
TTF = (2008.16667:0.083:2019.25)'
figure; 
for j=1:1
    subplot(1,1,j);
    h1= plot(TTF((end-30):end),A3{(end-30):end,j});
    hold on; 
    h2=plot(TTF(idxF),FY2_m6(:,j));
    h3=plot(TTF(idxF),YFI_m6(:,j,1),'k--');
    plot(TTF(idxF),YFI_m6(:,j,2),'k--');
    title('Model Adequacy: VAR best fitting model (6-Month Bond)');
   
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'southwest')
    legend('boxoff');
    set(gcf, 'color',  'w');
    ylim([-.4 1]);
    hold off;
end     

%Evaluation form ARIMA (Estimation Results)

RMSE_VAR1_est_m6=sqrt(mean(E1_m6(:,1).^2));
RMSE_VAR2_est_m6=sqrt(mean(E2_m6(:,1).^2));
RMSE_VAR3_est_m6=sqrt(mean(E3_m6(:,1).^2));

MAE1_VAR1_est_m6= mean(abs(E1_m6(:,1)));
MAE2_VAR2_est_m6= mean(abs(E2_m6(:,1)));
MAE3_VAR3_est_m6= mean(abs(E3_m6(:,1)));

acf_Error1_VAR1_est_m6 = autocorr(E1_m6(:,1));
acf_Error2_VAR2_est_m6 = autocorr(E2_m6(:,1));
acf_Error3_VAR3_est_m6 = autocorr(E3_m6(:,1));


%%%%%Sum-of-squares error between the predictors and the observed figures

Error1_m6=A3{idxF,:}-FY1_m6;
Error2_m6=A3{idxF,:}-FY2_m6;
Error3_m6=A3{idxF,:}-FY3_m6;

RMSEVAR1_m6=sqrt(mean(Error1_m6(:,1).^2))
RMSEVAR2_m6=sqrt(mean(Error2_m6(:,1).^2))
RMSEVAR3_m6=sqrt(mean(Error3_m6(:,1).^2))

MAE1_m6= mean(abs(Error1_m6(:,1)))
MAE2_m6= mean(abs(Error2_m6(:,1)))
MAE3_m6= mean(abs(Error3_m6(:,1)))

acf_Error1_m6 = autocorr(Error1_m6(:,1));
acf_Error2_m6 = autocorr(Error2_m6(:,1));
acf_Error3_m6 = autocorr(Error3_m6(:,1));



SSerror1_m6=Error1_m6(:)'*Error1_m6(:);
SSerror2_m6=Error2_m6(:)'*Error2_m6(:);
SSerror3_m6=Error1_m6(:)'*Error3_m6(:);

figure;
bar([SSerror1_m6 SSerror2_m6 SSerror3_m6], .5);
ylabel('Sum of squared error');
set(gca, 'XTickLabel', {'VAR1' 'VAR2' 'VAR3'});
title('Sum of Squared Forecast Errors');

summarize(EstMdl1_m6);


%%Forecast Observations
[YPred_m6, YCov_m6]=forecast(EstMdl2_m6,24,A3{10:134,:});

YFirst_m6 = A4(ixd2, {'m_6' 'rex' 'gdp' 'tiie28' 'oil' 'effr'});
YFirst_m6 = A4(121:135,:);

EndPt_m6 = YFirst_m6{end,:};
EndPt_m6(:,1:6) = log(EndPt_m6(:,1:6));
YPred_m6(:,1:6)=YPred_m6(:,1:6)/100;
YPred_m6=[EndPt_m6; YPred_m6];
YPred_m6(:,1:6)=cumsum(YPred_m6(:,1:6));
YPred_m6(:,1:6)=exp(YPred_m6(:,1:6));
TTF2= (2019.25:0.083:2021.25)'
TTT3= (2008.083:0.083:2019.25)'
%%%%%
%--------------
 
    plot(TTF2,YPred_m6(:,1),'--b');
    hold on 
    plot(TTT3(121:135),YFirst_m6{:,1},'k');
    title(EstMdl2_m6.SeriesNames{1});
    h=gca;
    fill([TTF2(1) h.XLim([2 2]) TTF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    title('VAR Forecast 24-months ahead (6-Month Bond) ') 
    set(gcf, 'color',  'w');
    hold off
    ylim([7.6 8.6]);
    
 %-------------------------- 1-Year Bond --------
 
 A5 = table(dy1,rexg,gdpg,tiie28g,oilg,effrg);
A6 = table(y1,rex,gdp,tiie28,oil,effr);
ixd=all(~ismissing(A5),2);
ixd2=all(~ismissing(A6),2);
A5=A5(ixd,:);
A6=A6(ixd2,:);
idxPre=16:134;
TTT=ceil(.82*size(A5,1));
TTTT_1=ceil(.82*size(A6,1));
idxEst=16:TTT;
idxEst_1=16:TTTT_1;
idxF=(TTT+1):size(A5,1);
idxF_2=(TTT+1):size(A6,1);


 %%% Select and Fit the Model
 

numseries=6;
seriesnam_y1={'1-Year Bond ','Mexico: Real Exchange Rate', 'Mexico: Global Economic Indicator', 'Mexico: Central Bank Funds Rate','US: Oil Price', 'US: Effective Federal Funds Rate'  };
VAR1f_y1=varm(numseries,1);
VAR1f_y1.SeriesNames=seriesnam_y1;
VAR2f_y1=varm(numseries,2);
VAR2f_y1.SeriesNames=seriesnam_y1;
VAR3f_y1=varm(numseries,3);
VAR3f_y1.SeriesNames=seriesnam_y1;

[EstMdl1_y1,EstSE1_y1,logL1_y1,E1_y1]= estimate(VAR1f_y1,A5{idxEst,:}, 'Y0',A5{idxPre,:});
[EstMdl2_y1,EstSE2_y1,logL2_y1,E2_y1]= estimate(VAR2f_y1,A5{idxEst,:}, 'Y0',A5{idxPre,:});
[EstMdl3_y1,EstSE3_y1,logL3_y1,E3_y1]= estimate(VAR3f_y1,A5{idxEst,:}, 'Y0',A5{idxPre,:});

%%Check Model Adequacy
EstMdl1_y1.Description
EstMdl2_y1.Description
EstMdl3_y1.Description

results1_y1=summarize(EstMdl1_y1);
np1_y1=results1_y1.NumEstimatedParameters;
results2_y1=summarize(EstMdl2_y1);
np2_y1=results2_y1.NumEstimatedParameters;
results3_y1=summarize(EstMdl3_y1);
np3_y1=results3_y1.NumEstimatedParameters;
%%Likelihood Ratio test of model specification 
[h_y1,pValue_y1,stat_y1,cValue_y1]=lratiotest(logL2_y1,logL1_y1,np2_y1- np1_y1)
[h1_y1,pValue1_y1,stat1_y1,cValue1_y1]=lratiotest(logL3_y1,logL1_y1,np3_y1- np1_y1)
%%Missing a matrix which store the information for the LM test

%%%Akaike Information Criterion (AIC)

AIC_y1=aicbic([logL1_y1 logL2_y1 logL3_y1], [np1_y1 np2_y1 np3_y1]);

%For the 3-Month the best model id the VAR1

%%Comparing the predictions 

[FY1_y1,FYCov1_y1]=forecast(EstMdl1_y1,fh,A5{idxEst,:});
[FY2_y1,FYCov2_y1]=forecast(EstMdl2_y1,fh,A5{idxEst,:});
[FY3_y1,FYCov3_y1]=forecast(EstMdl3_y1,fh,A5{idxEst,:});

%Estimating the 95% forecast interval for the best fitting model

extractMSE=@(x)diag(x)';
MSE_y1=cellfun(extractMSE, FYCov1_y1, 'UniformOutput', false);
SE_y1=sqrt(cell2mat(MSE_y1));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% Change
YFI_y1= zeros(fh,EstMdl1_y1.NumSeries,2);
YFI_y1(:,:,1) = FY1_y1 - 2*SE_y1; %Lower Band
YFI_y1(:,:,2) = FY1_y1 + 2*SE_y1; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


%%plotting the results
TTF = (2008.16667:0.083:2019.25)'
figure; 
for j=1:1
    subplot(1,1,j);
    h1= plot(TTF((end-30):end),A5{(end-30):end,j});
    hold on; 
    h2=plot(TTF(idxF),FY1_y1(:,j));
    h3=plot(TTF(idxF),YFI_y1(:,j,1),'k--');
    plot(TTF(idxF),YFI_y1(:,j,2),'k--');
    title('Model Adequacy: VAR best fitting model (1-Year Bond)');
   
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'southwest')
    legend('boxoff');
    set(gcf, 'color',  'w');
     ylim([-.4 1.1]);
    hold off;
end     

%Evaluation form ARIMA (Estimation Results)

RMSE_VAR1_est_y1=sqrt(mean(E1_y1(:,1).^2));
RMSE_VAR2_est_y1=sqrt(mean(E2_y1(:,1).^2));
RMSE_VAR3_est_y1=sqrt(mean(E3_y1(:,1).^2));

MAE1_VAR1_est_y1= mean(abs(E1_y1(:,1)));
MAE2_VAR2_est_y1= mean(abs(E2_y1(:,1)));
MAE3_VAR3_est_y1= mean(abs(E3_y1(:,1)));

acf_Error1_VAR1_est_y1 = autocorr(E1_y1(:,1));
acf_Error2_VAR2_est_y1 = autocorr(E2_y1(:,1));
acf_Error3_VAR3_est_y1 = autocorr(E3_y1(:,1));

%%%%%Sum-of-squares error between the predictors and the observed figures

Error1_y1=A5{idxF,:}-FY1_y1;
Error2_y1=A5{idxF,:}-FY2_y1;
Error3_y1=A5{idxF,:}-FY3_y1;

RMSEVAR1_y1=sqrt(mean(Error1_y1(:,1).^2))
RMSEVAR2_y1=sqrt(mean(Error2_y1(:,1).^2))
RMSEVAR3_y1=sqrt(mean(Error3_y1(:,1).^2))

MAE1_y1= mean(abs(Error1_y1(:,1)))
MAE2_y1= mean(abs(Error2_y1(:,1)))
MAE3_y1= mean(abs(Error3_y1(:,1)))

acf_Error1_y1 = autocorr(Error1_y1(:,1));
acf_Error2_y1 = autocorr(Error2_y1(:,1));
acf_Error3_y1 = autocorr(Error3_y1(:,1));


%The test indicate that the VAR2 model is the best for the Forecast

SSerror1_y1=Error1_y1(:)'*Error1_y1(:);
SSerror2_y1=Error2_y1(:)'*Error2_y1(:);
SSerror3_y1=Error1_y1(:)'*Error3_y1(:);

figure;
bar([SSerror1_y1 SSerror2_y1 SSerror3_y1], .5);
ylabel('Sum of squared error');
set(gca, 'XTickLabel', {'VAR1' 'VAR2' 'VAR3'});
title('Sum of Squared Forecast Errors');

summarize(EstMdl1_y1);

%%Forecast Observations
[YPred_y1, YCov_y1]=forecast(EstMdl2_y1,24,A5{10:134,:});

YFirst_y1 = A6(ixd2, {'y1' 'rex' 'gdp' 'tiie28' 'oil' 'effr'});
YFirst_y1 = A6(120:135,:);

EndPt_y1 = YFirst_y1{end,:};
EndPt_y1(:,1:6) = log(EndPt_y1(:,1:6));
YPred_y1(:,1:6)=YPred_y1(:,1:6)/100;
YPred_y1=[EndPt_y1; YPred_y1];
YPred_y1(:,1:6)=cumsum(YPred_y1(:,1:6));
YPred_y1(:,1:6)=exp(YPred_y1(:,1:6));
TTF2= (2019.25:0.083:2021.25)'
TTT3= (2008.083:0.083:2019.25)'
%%%%%
%--------------
 
    plot(TTF2,YPred_y1(:,1),'--b');
    hold on 
    plot(TTT3(120:135),YFirst_y1{:,1},'k');
    title(EstMdl2_y1.SeriesNames{1});
    h=gca;
    fill([TTF2(1) h.XLim([2 2]) TTF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    title('VAR Forecast 24-months ahead (1-Year Bond)') 
    set(gcf, 'color',  'w');
    hold off
    ylim([7.4 8.7]);
 
%------------------- 5-YEar Bond
 
A7 = table(dy5,rexg,gdpg,tiie28g,oilg,effrg);
A8 = table(y5,rex,gdp,tiie28,oil,effr);
ixd=all(~ismissing(A7),2);
ixd2=all(~ismissing(A8),2);
A7=A7(ixd,:);
A8=A8(ixd2,:);
idxPre=16:134;
TTT=ceil(.82*size(A7,1));
TTTT_1=ceil(.82*size(A8,1));
idxEst=16:TTT;
idxEst_1=16:TTTT_1;
idxF=(TTT+1):size(A7,1);
idxF_2=(TTT+1):size(A8,1);

 %%% Select and Fit the Model
 
numseries=6;
seriesnam_y5={'5-Year Bond ','Mexico: Real Exchange Rate', 'Mexico: Global Economic Indicator', 'Mexico: Central Bank Funds Rate','US: Oil Price', 'US: Effective Federal Funds Rate'  };
VAR1f_y5=varm(numseries,1);
VAR1f_y5.SeriesNames=seriesnam_y5;
VAR2f_y5=varm(numseries,2);
VAR2f_y5.SeriesNames=seriesnam_y5;
VAR3f_y5=varm(numseries,3);
VAR3f_y5.SeriesNames=seriesnam_y5;

[EstMdl1_y5,EstSE1_y5,logL1_y5,E1_y5]= estimate(VAR1f_y5,A7{idxEst,:}, 'Y0',A7{idxPre,:});
[EstMdl2_y5,EstSE2_y5,logL2_y5,E2_y5]= estimate(VAR2f_y5,A7{idxEst,:}, 'Y0',A7{idxPre,:});
[EstMdl3_y5,EstSE3_y5,logL3_y5,E3_y5]= estimate(VAR3f_y5,A7{idxEst,:}, 'Y0',A7{idxPre,:});

%%Check Model Adequacy
EstMdl1_y5.Description
EstMdl2_y5.Description
EstMdl3_y5.Description

results1_y5=summarize(EstMdl1_y5);
np1_y5=results1_y5.NumEstimatedParameters;
results2_y5=summarize(EstMdl2_y5);
np2_y5=results2_y5.NumEstimatedParameters;
results3_y5=summarize(EstMdl3_y5);
np3_y5=results3_y5.NumEstimatedParameters;
%%Likelihood Ratio test of model specification 
[h_y5,pValue_y5,stat_y5,cValue_y5]=lratiotest(logL2_y5,logL1_y5,np2_y5- np1_y5)
[h1_y5,pValue1_y5,stat1_y5,cValue1_y5]=lratiotest(logL3_y5,logL1_y5,np3_y5- np1_y5)
%%Missing a matrix which store the information for the LM test

%%%Akaike Information Criterion (AIC)

AIC_y5=aicbic([logL1_y5 logL2_y5 logL3_y5], [np1_y5 np2_y5 np3_y5]);

%For the 3-Month the best model id the VAR1

%%Comparing the predictions 

[FY1_y5,FYCov1_y5]=forecast(EstMdl1_y5,fh,A7{idxEst,:});
[FY2_y5,FYCov2_y5]=forecast(EstMdl2_y5,fh,A7{idxEst,:});
[FY3_y5,FYCov3_y5]=forecast(EstMdl3_y5,fh,A7{idxEst,:});

%Estimating the 95% forecast interval for the best fitting model

extractMSE=@(x)diag(x)';
MSE_y5=cellfun(extractMSE, FYCov2_y5, 'UniformOutput', false);
SE_y5=sqrt(cell2mat(MSE_y5));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% Change
YFI_y5= zeros(fh,EstMdl2_y5.NumSeries,2);
YFI_y5(:,:,1) = FY1_y5 - 2*SE_y5; %Lower Band
YFI_y5(:,:,2) = FY1_y5 + 2*SE_y5; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


%%plotting the results
TTF = (2008.16667:0.083:2019.25)'
figure; 
for j=1:1
    subplot(1,1,j);
    h1= plot(TTF((end-30):end),A7{(end-30):end,j});
    hold on; 
    h2=plot(TTF(idxF),FY1_y5(:,j));
    h3=plot(TTF(idxF),YFI_y5(:,j,1),'k--');
    plot(TTF(idxF),YFI_y5(:,j,2),'k--');
    title('Model Adequacy: VAR best fitting model (5-Year Bond)');
   
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'southwest')
    legend('boxoff');
    set(gcf, 'color',  'w');
     ylim([-.8 1.1]);
    hold off;
end     

%Evaluation form ARIMA (Estimation Results)

RMSE_VAR1_est_y5=sqrt(mean(E1_y5(:,1).^2));
RMSE_VAR2_est_y5=sqrt(mean(E2_y5(:,1).^2));
RMSE_VAR3_est_y5=sqrt(mean(E3_y5(:,1).^2));

MAE1_VAR1_est_y5= mean(abs(E1_y5(:,1)));
MAE2_VAR2_est_y5= mean(abs(E2_y5(:,1)));
MAE3_VAR3_est_y5= mean(abs(E3_y5(:,1)));

acf_Error1_VAR1_est_y5 = autocorr(E1_y5(:,1));
acf_Error2_VAR2_est_y5 = autocorr(E2_y5(:,1));
acf_Error3_VAR3_est_y5 = autocorr(E3_y5(:,1));


%%%%%Sum-of-squares error between the predictors and the observed figures

Error1_y5=A7{idxF,:}-FY1_y5;
Error2_y5=A7{idxF,:}-FY2_y5;
Error3_y5=A7{idxF,:}-FY3_y5;

RMSEVAR1_y5=sqrt(mean(Error1_y5(:,1).^2))
RMSEVAR2_y5=sqrt(mean(Error2_y5(:,1).^2))
RMSEVAR3_y5=sqrt(mean(Error3_y5(:,1).^2))

MAE1_y5= mean(abs(Error1_y5(:,1)))
MAE2_y5= mean(abs(Error2_y5(:,1)))
MAE3_y5= mean(abs(Error3_y5(:,1)))

acf_Error1_y5 = autocorr(Error1_y5(:,1));
acf_Error2_y5 = autocorr(Error2_y5(:,1));
acf_Error3_y5 = autocorr(Error3_y5(:,1));

%The test indicate that the VAR2 model is the best for the Forecast

SSerror1_y5=Error1_y5(:)'*Error1_y5(:);
SSerror2_y5=Error2_y5(:)'*Error2_y5(:);
SSerror3_y5=Error1_y5(:)'*Error3_y5(:);

figure;
bar([SSerror1_y5 SSerror2_y5 SSerror3_y5], .5);
ylabel('Sum of squared error');
set(gca, 'XTickLabel', {'VAR1' 'VAR2' 'VAR3'});
title('Sum of Squared Forecast Errors');

summarize(EstMdl1_y5);


%%Forecast Observations
[YPred_y5, YCov_y5]=forecast(EstMdl2_y5,24,A7{10:134,:});

YFirst_y5 = A8(ixd2, {'y5' 'rex' 'gdp' 'tiie28' 'oil' 'effr'});
YFirst_y5 = A8(120:135,:);

EndPt_y5 = YFirst_y5{end,:};
EndPt_y5(:,1:6) = log(EndPt_y5(:,1:6));
YPred_y5(:,1:6)=YPred_y5(:,1:6)/100;
YPred_y5=[EndPt_y5; YPred_y5];
YPred_y5(:,1:6)=cumsum(YPred_y5(:,1:6));
YPred_y5(:,1:6)=exp(YPred_y5(:,1:6));
TTF2= (2019.25:0.083:2021.25)'
TTT3= (2008.083:0.083:2019.25)'
%%%%%
%--------------
 
    plot(TTF2,YPred_y5(:,1),'--b');
    hold on 
    plot(TTT3(120:135),YFirst_y5{:,1},'k');
    title(EstMdl2_y5.SeriesNames{1});
    h=gca;
    fill([TTF2(1) h.XLim([2 2]) TTF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    title('VAR Forecast 24-months ahead (5-Year Bond)') 
    set(gcf, 'color',  'w');
    hold off
    ylim([7 9]);
  
%-----------10-Year Bond 
 
A9 = table(dy10,rexg,gdpg,tiie28g,oilg,effrg);
A10 = table(y10,rex,gdp,tiie28,oil,effr);
ixd=all(~ismissing(A9),2);
ixd2=all(~ismissing(A10),2);
A9=A9(ixd,:);
A10=A10(ixd2,:);
idxPre=16:134;
TTT=ceil(.82*size(A9,1));
TTTT_1=ceil(.82*size(A10,1));
idxEst=16:TTT;
idxEst_1=16:TTTT_1;
idxF=(TTT+1):size(A9,1);
idxF_2=(TTT+1):size(A10,1);

 %%% Select and Fit the Model
 
numseries=6;
seriesnam_y10={'10-Year Bond ','Mexico: Real Exchange Rate', 'Mexico: Global Economic Indicator', 'Mexico: Central Bank Funds Rate','US: Oil Price', 'US: Effective Federal Funds Rate'  };
VAR1f_y10=varm(numseries,1);
VAR1f_y10.SeriesNames=seriesnam_y10;
VAR2f_y10=varm(numseries,2);
VAR2f_y10.SeriesNames=seriesnam_y10;
VAR3f_y10=varm(numseries,3);
VAR3f_y10.SeriesNames=seriesnam_y10;

[EstMdl1_y10,EstSE1_y10,logL1_y10,E1_y10]= estimate(VAR1f_y10,A9{idxEst,:}, 'Y0',A9{idxPre,:});
[EstMdl2_y10,EstSE2_y10,logL2_y10,E2_y10]= estimate(VAR2f_y10,A9{idxEst,:}, 'Y0',A9{idxPre,:});
[EstMdl3_y10,EstSE3_y10,logL3_y10,E3_y10]= estimate(VAR3f_y10,A9{idxEst,:}, 'Y0',A9{idxPre,:});

%%Check Model Adequacy
EstMdl1_y10.Description
EstMdl2_y10.Description
EstMdl3_y10.Description

results1_y10=summarize(EstMdl1_y10);
np1_y10=results1_y10.NumEstimatedParameters;
results2_y10=summarize(EstMdl2_y10);
np2_y10=results2_y10.NumEstimatedParameters;
results3_y10=summarize(EstMdl3_y10);
np3_y10=results3_y10.NumEstimatedParameters;
%%Likelihood Ratio test of model specification 
[h_y10,pValue_y10,stat_y10,cValue_y10]=lratiotest(logL2_y10,logL1_y10,np2_y10- np1_y10)
[h1_y10,pValue1_y10,stat1_y10,cValue1_y10]=lratiotest(logL3_y10,logL1_y10,np3_y10- np1_y10)
%%Missing a matrix which store the information for the LM test

%%%Akaike Information Criterion (AIC)

AIC_y10=aicbic([logL1_y10 logL2_y10 logL3_y10], [np1_y10 np2_y10 np3_y10]);

%For the 3-Month the best model id the VAR1

%%Comparing the predictions 

[FY1_y10,FYCov1_y10]=forecast(EstMdl1_y10,fh,A9{idxEst,:});
[FY2_y10,FYCov2_y10]=forecast(EstMdl2_y10,fh,A9{idxEst,:});
[FY3_y10,FYCov3_y10]=forecast(EstMdl3_y10,fh,A9{idxEst,:});

%Estimating the 95% forecast interval for the best fitting model

extractMSE=@(x)diag(x)';
MSE_y10=cellfun(extractMSE, FYCov2_y10, 'UniformOutput', false);
SE_y10=sqrt(cell2mat(MSE_y10));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% Change
YFI_y10= zeros(fh,EstMdl2_y10.NumSeries,2);
YFI_y10(:,:,1) = FY1_y10 - 2*SE_y10; %Lower Band
YFI_y10(:,:,2) = FY1_y10 + 2*SE_y10; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


%%plotting the results
TTF = (2008.16667:0.083:2019.25)'
figure; 
for j=1:1
    subplot(1,1,j);
    h1= plot(TTF((end-30):end),A9{(end-30):end,j});
    hold on; 
    h2=plot(TTF(idxF),FY1_y10(:,j));
    h3=plot(TTF(idxF),YFI_y10(:,j,1),'k--');
    plot(TTF(idxF),YFI_y10(:,j,2),'k--');
    title('Model Adequacy: VAR best fitting model (10-Year Bond)');
   
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'southwest')
    legend('boxoff');
    set(gcf, 'color',  'w');
    hold off;
end   
%Evaluation form ARIMA (Estimation Results)

RMSE_VAR1_est_y10=sqrt(mean(E1_y10(:,1).^2));
RMSE_VAR2_est_y10=sqrt(mean(E2_y10(:,1).^2));
RMSE_VAR3_est_y10=sqrt(mean(E3_y10(:,1).^2));

MAE1_VAR1_est_y10= mean(abs(E1_y10(:,1)));
MAE2_VAR2_est_y10= mean(abs(E2_y10(:,1)));
MAE3_VAR3_est_y10= mean(abs(E3_y10(:,1)));

acf_Error1_VAR1_est_y10 = autocorr(E1_y10(:,1));
acf_Error2_VAR2_est_y10 = autocorr(E2_y10(:,1));
acf_Error3_VAR3_est_y10 = autocorr(E3_y10(:,1));


%%%%%Sum-of-squares error between the predictors and the observed figures

Error1_y10=A9{idxF,:}-FY1_y10;
Error2_y10=A9{idxF,:}-FY2_y10;
Error3_y10=A9{idxF,:}-FY3_y10;

RMSEVAR1_y10=sqrt(mean(Error1_y10(:,1).^2))
RMSEVAR2_y10=sqrt(mean(Error2_y10(:,1).^2))
RMSEVAR3_y10=sqrt(mean(Error3_y10(:,1).^2))

MAE1_y10= mean(abs(Error1_y10(:,1)))
MAE2_y10= mean(abs(Error2_y10(:,1)))
MAE3_y10= mean(abs(Error3_y10(:,1)))

acf_Error1_y10 = autocorr(Error1_y10(:,1));
acf_Error2_y10 = autocorr(Error2_y10(:,1));
acf_Error3_y10 = autocorr(Error3_y10(:,1));

%The test indicate that the VAR2 model is the best for the Forecast

SSerror1_y10=Error1_y10(:)'*Error1_y10(:);
SSerror2_y10=Error2_y10(:)'*Error2_y10(:);
SSerror3_y10=Error1_y10(:)'*Error3_y10(:);

figure;
bar([SSerror1_y10 SSerror2_y10 SSerror3_y10], .5);
ylabel('Sum of squared error');
set(gca, 'XTickLabel', {'VAR1' 'VAR2' 'VAR3'});
title('Sum of Squared Forecast Errors');

summarize(EstMdl1_y10);


%%Forecast Observations
[YPred_y10, YCov_y10]=forecast(EstMdl2_y10,24,A9{10:134,:});

YFirst_y10 = A10(ixd2, {'y10' 'rex' 'gdp' 'tiie28' 'oil' 'effr'});
YFirst_y10 = A10(120:135,:);

EndPt_y10 = YFirst_y10{end,:};
EndPt_y10(:,1:6) = log(EndPt_y10(:,1:6));
YPred_y10(:,1:6)=YPred_y10(:,1:6)/100;
YPred_y10=[EndPt_y10; YPred_y10];
YPred_y10(:,1:6)=cumsum(YPred_y10(:,1:6));
YPred_y10(:,1:6)=exp(YPred_y10(:,1:6));
TTF2= (2019.25:0.083:2021.25)'
TTT3= (2008.083:0.083:2019.25)'
%%%%%
%--------------
 
    plot(TTF2,YPred_y10(:,1),'--b');
    hold on 
    plot(TTT3(120:135),YFirst_y10{:,1},'k');
    title(EstMdl2_y10.SeriesNames{1});
    h=gca;
    fill([TTF2(1) h.XLim([2 2]) TTF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    title('VAR Forecast 24-months ahead (10-Year Bond)') 
    set(gcf, 'color',  'w');
    hold off
    ylim([7 9.4]);
    
    
%----------------30-Year Bond

A11 = table(dy30,rexg,gdpg,tiie28g,oilg,effrg);
A12 = table(y30,rex,gdp,tiie28,oil,effr);
ixd=all(~ismissing(A11),2);
ixd2=all(~ismissing(A12),2);
A11=A11(ixd,:);
A12=A12(ixd2,:);
idxPre=16:134;
TTT=ceil(.82*size(A11,1));
TTTT_1=ceil(.82*size(A12,1));
idxEst=16:TTT;
idxEst_1=16:TTTT_1;
idxF=(TTT+1):size(A11,1);
idxF_2=(TTT+1):size(A12,1);


 %%% Select and Fit the Model
 

numseries=6;
seriesnam_y30={'30-Year Bond ', 'Mexico: Real Exchange Rate', 'Mexico: Global Economic Indicator', 'Mexico: Central Bank Funds Rate','US: Oil Price', 'US: Effective Federal Funds Rate'  };
VAR1f_y30=varm(numseries,1);
VAR1f_y30.SeriesNames=seriesnam_y30;
VAR2f_y30=varm(numseries,2);
VAR2f_y30.SeriesNames=seriesnam_y30;
VAR3f_y30=varm(numseries,3);
VAR3f_y30.SeriesNames=seriesnam_y30;

[EstMdl1_y30,EstSE1_y30,logL1_y30,E1_y30]= estimate(VAR1f_y30,A11{idxEst,:}, 'Y0',A11{idxPre,:});
[EstMdl2_y30,EstSE2_y30,logL2_y30,E2_y30]= estimate(VAR2f_y30,A11{idxEst,:}, 'Y0',A11{idxPre,:});
[EstMdl3_y30,EstSE3_y30,logL3_y30,E3_y30]= estimate(VAR3f_y30,A11{idxEst,:}, 'Y0',A11{idxPre,:});

%%Check Model Adequacy
EstMdl1_y30.Description
EstMdl2_y30.Description
EstMdl3_y30.Description

results1_y30=summarize(EstMdl1_y30);
np1_y30=results1_y30.NumEstimatedParameters;
results2_y30=summarize(EstMdl2_y30);
np2_y30=results2_y30.NumEstimatedParameters;
results3_y30=summarize(EstMdl3_y30);
np3_y30=results3_y30.NumEstimatedParameters;
%%Likelihood Ratio test of model specification 
[h_y30,pValue_y30,stat_y30,cValue_y30]=lratiotest(logL2_y30,logL1_y30,np2_y30- np1_y30);
[h1_y30,pValue1_y30,stat1_y30,cValue1_y30]=lratiotest(logL3_y30,logL1_y30,np3_y30- np1_y30);
%%Missing a matrix which store the information for the LM test

%%%Akaike Information Criterion (AIC)

AIC_y30=aicbic([logL1_y30 logL2_y30 logL3_y30], [np1_y30 np2_y30 np3_y30]);

%For the 3-Month the best model id the VAR1

%%Comparing the predictions 

[FY1_y30,FYCov1_y30]=forecast(EstMdl1_y30,fh,A11{idxEst,:});
[FY2_y30,FYCov2_y30]=forecast(EstMdl2_y30,fh,A11{idxEst,:});
[FY3_y30,FYCov3_y30]=forecast(EstMdl3_y30,fh,A11{idxEst,:});

%Estimating the 95% forecast interval for the best fitting model

extractMSE=@(x)diag(x)';
MSE_y30=cellfun(extractMSE, FYCov2_y30, 'UniformOutput', false);
SE_y30=sqrt(cell2mat(MSE_y30));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% Change
YFI_y30= zeros(fh,EstMdl2_y30.NumSeries,2);
YFI_y30(:,:,1) = FY1_y30 - 2*SE_y30; %Lower Band
YFI_y30(:,:,2) = FY1_y30 + 2*SE_y30; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


%%plotting the results
TTF = (2008.16667:0.083:2019.25)'
figure; 
for j=1:1
    subplot(1,1,j);
    h1= plot(TTF((end-30):end),A11{(end-30):end,j});
    hold on; 
    h2=plot(TTF(idxF),FY1_y30(:,j));
    h3=plot(TTF(idxF),YFI_y30(:,j,1),'k--');
    plot(TTF(idxF),YFI_y30(:,j,2),'k--');
    title('Model Adequacy: VAR best fitting model (30-Year Bond)');
   
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'southwest')
    legend('boxoff');
    set(gcf, 'color',  'w');
    hold off;
end     

%Evaluation form ARIMA (Estimation Results)

RMSE_VAR1_est_y30=sqrt(mean(E1_y30(:,1).^2));
RMSE_VAR2_est_y30=sqrt(mean(E2_y30(:,1).^2));
RMSE_VAR3_est_y30=sqrt(mean(E3_y30(:,1).^2));

MAE1_VAR1_est_y30= mean(abs(E1_y30(:,1)));
MAE2_VAR2_est_y30= mean(abs(E2_y30(:,1)));
MAE3_VAR3_est_y30= mean(abs(E3_y30(:,1)));

acf_Error1_VAR1_est_y30 = autocorr(E1_y30(:,1));
acf_Error2_VAR2_est_y30 = autocorr(E2_y30(:,1));
acf_Error3_VAR3_est_y30 = autocorr(E3_y30(:,1));


%%%%%Sum-of-squares error between the predictors and the observed figures

Error1_y30=A11{idxF,:}-FY1_y30;
Error2_y30=A11{idxF,:}-FY2_y30;
Error3_y30=A11{idxF,:}-FY3_y30;

RMSEVAR1_y30=sqrt(mean(Error1_y30(:,1).^2));
RMSEVAR2_y30=sqrt(mean(Error2_y30(:,1).^2));
RMSEVAR3_y30=sqrt(mean(Error3_y30(:,1).^2));

MAE1_y30= mean(abs(Error1_y30(:,1)))
MAE2_y30= mean(abs(Error2_y30(:,1)))
MAE3_y30= mean(abs(Error3_y30(:,1)))

acf_Error1_y30 = autocorr(Error1_y30(:,1));
acf_Error2_y30 = autocorr(Error2_y30(:,1));
acf_Error3_y30 = autocorr(Error3_y30(:,1));

%The test indicate that the VAR2 model is the best for the Forecast

SSerror1_y30=Error1_y30(:)'*Error1_y30(:);
SSerror2_y30=Error2_y30(:)'*Error2_y30(:);
SSerror3_y30=Error1_y30(:)'*Error3_y30(:);

figure;
bar([SSerror1_y30 SSerror2_y30 SSerror3_y30], .5);
ylabel('Sum of squared error');
set(gca, 'XTickLabel', {'VAR1' 'VAR2' 'VAR3'});
title('Sum of Squared Forecast Errors');

summarize(EstMdl1_y30);

%%Forecast Observations
[YPred_y30, YCov_y30]=forecast(EstMdl2_y30,24,A11{10:134,:});

YFirst_y30 = A12(ixd2, {'y30' 'rex' 'gdp' 'tiie28' 'oil' 'effr'});
YFirst_y30 = A12(120:135,:);

EndPt_y30 = YFirst_y30{end,:};
EndPt_y30(:,1:6) = log(EndPt_y30(:,1:6));
YPred_y30(:,1:6)=YPred_y30(:,1:6)/100;
YPred_y30=[EndPt_y30; YPred_y30];
YPred_y30(:,1:6)=cumsum(YPred_y30(:,1:6));
YPred_y30(:,1:6)=exp(YPred_y30(:,1:6));
TTF2= (2019.25:0.083:2021.25)'
TTT3= (2008.083:0.083:2019.25)'
%%%%%
%--------------
 
    plot(TTF2,YPred_y30(:,1),'--b');
    hold on 
    plot(TTT3(120:135),YFirst_y30{:,1},'k');
    title(EstMdl2_y30.SeriesNames{1});
    h=gca;
    fill([TTF2(1) h.XLim([2 2]) TTF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    title('VAR Forecast 24-months ahead (30-Year Bond)') 
    set(gcf, 'color',  'w');
    hold off
    ylim([7.5 9.8]);

% Deibod Mariano Test 

DM_m3 = dmtest(Error_m3_aic, Error1(:,1), 1);
DM_m3_2 = dmtest(Error_m3_aic, Error1(:,1), 2);
DM_m6 = dmtest(Error_m6_aic, Error1_m6(:,1), 1);
DM_m6_2 = dmtest(Error_m6_aic, Error1_m6(:,1), 2);
DM_y1 = dmtest(Error_y1_aic, Error1_y1(:,1), 1);
DM_y1_2 = dmtest(Error_y1_aic, Error1_y1(:,1), 2);
DM_y5 = dmtest(Error_aic_y5, Error1_y5(:,1), 1);
DM_y5_2 = dmtest(Error_aic_y5, Error1_y5(:,1), 2);
DM_y10 = dmtest(Error_aic_y10, Error1_y10(:,1), 1);
DM_y10_2 = dmtest(Error_aic_y10, Error1_y10(:,1), 2);
DM_y30 = dmtest(Error_aic_y30, Error1_y30(:,1), 1);
DM_y30_2 = dmtest(Error_aic_y30, Error1_y30(:,1), 2);


%-------------Ploting the Yield Curve 
 
yield_plot_var= [YPred(2,1) YPred_m6(2,1) YPred_y1(2,1) YPred_y5(2,1) YPred_y10(2,1) YPred_y30(2,1); YPred(25,1) YPred_m6(25,1) YPred_y1(25,1) YPred_y5(25,1) YPred_y10(25,1) YPred_y30(25,1); YPred_arima(2) YPred_arima_m6(2) YPred_arima_y1(2) YPred_arima_y5(2) YPred_arima_y10(2) YPred_arima_y30(2); YPred_arima(25) YPred_arima_m6(25) YPred_arima_y1(25) YPred_arima_y5(25) YPred_arima_y10(25) YPred_arima_y30(25)];
s1_var = categorical ({'3-Month' '6-Month' '1-Year' '5-Year' '10-Year' '30-Year'});
figure;
plot(s1_var,yield_plot_var, 'LineWidth', 1.9) , title('Graph 4: Forecast Government Yield Curve', 'FontSize', 14), , legend({'VAR model: April 2019',' VAR model: April 2020', 'ARIMA model: April 2019', 'ARIMA model: April 2020' }, 'location','best'), legend('boxoff'), xlabel({'Source:Bloomberg'}), set(gca,'FontSize',12), set(gcf, 'color',  'w'), xlabel('Maturity');
ax=gca;
ax.XAxis.Categories;
ax.XAxis.Categories = {'3-Month', '6-Month', '1-Year', '5-Year', '10-Year', '30-Year'};
ylim([7.7 8.5]);




