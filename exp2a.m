wp=0.45*pi; %Passband edge 
ws=0.55*pi; %Stopband edge

F=[0 wp/pi ws/pi 1]; %Normalised frequencies
A=[1 1 0 0];         %Desired amplitude response (1 in passband, 0 in stopband)
W=[1 1];

%Filter order, odd for type 2
N=55;

%Designing our FIR Type-2 filter
h=firpm(N, F, A, W);

%M=2 for 2 polyphase decomposition
M=2;

%Splitting the filter coefficients into polyphase components
H0_0 = h(1:2:end); %Coefficients for H°0
H0_1 = h(2:2:end);  %Coefficients for H°1 

% Phase-shifted version of the filter
h_shifted = h.*real(exp(1j*-1*pi*(0:N))); % Apply a -π phase shift which corresponds to H°(-z)
zerophase(dfilt.dffir(0.5*(conv(h, h) - conv(h_shifted,h_shifted))))

[x,fs]=audioread('music16khz.wav');

xd=downsample([x;0], 2); %[x;0] will append a zero to the signal x so x and s have the same length
sd=downsample([0;x], 2); %[0;x] adds a zero at the beginning of the signal, inducing a time delay of 1

t0=filter(H0_0,1,xd); %output of H°0 (0th polyphase filter)
t1=filter(H0_1,1,sd); %output of H°1 (1st polyphase filter)

%final analysis filterbank outputs after passing through matrix W*
vd0=t0+t1;
vd1=t0-t1;

%plotting vd0 and vd1
N_vd0 = length(vd0);
f_vd0 = linspace(-pi, pi, N_vd0); %Frequency vector 
Vd0 = fft(vd0);  %FFT of vd0
Vd0_magnitude = abs(fftshift(Vd0));%Magnitude and shift FFT 

N_vd1 = length(vd1);
f_vd1 = linspace(-pi, pi, N_vd1); %Frequency vector 
Vd1 = fft(vd1);  %FFT of vd1
Vd1_magnitude = abs(fftshift(Vd1));%Magnitude and shift FFT

%Plotting the magnitude spectrum of vd0 and vd1
figure;

%Plotting vd0 spectrum
subplot(2, 1, 1);
plot(f_vd0, Vd0_magnitude);
title('Magnitude Spectrum of vd0');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xlim([-pi pi]);
xticks(-pi:pi/2:pi); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
grid on;

% Plotting vd1 spectrum
subplot(2, 1, 2);
plot(f_vd1, Vd1_magnitude);
title('Magnitude Spectrum of vd1');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xlim([-pi pi]);
xticks(-pi:pi/2:pi); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
grid on;


%passing vd0 and vd1 through W to obtain inputs to K0 and K1

i0=vd0+vd1;
i1=vd0-vd1;

%Case 1

o1=filter(H0_1,1,i0); %K0=H0_1
o2=filter(H0_0,1,i1); %K1=H0_0

comm_o1=upsample(o1,2);
comm_o2=upsample(o2,2);
comm_o1=[0;comm_o1]; %We delay the first output by 1
comm_o2=[comm_o2;0]; %We append the second output with a 0 to make the vectors equal in size
y1=comm_o1+comm_o2;%the final output is the sum of these two outputs

%Plotting magnitude spectrum of the original input signal
N_input = length(x);  %Length of the input signal
f_input = linspace(-pi, pi, N_input);  %Frequency vector from -pi to pi for the input signal

X_input = fft(x);  %Computing FFT of input signal
X_input_magnitude = abs(fftshift(X_input));  %Magnitude and shift FFT 

%Plotting magnitude spectrum of the reconstructed signal y1
N_y1 = length(y1); 
f_y1 = linspace(-pi, pi, N_y1); 

Y1 = fft(y1); 
Y1_magnitude = abs(fftshift(Y1));  

% Plotting the magnitude spectrum of both input signal and y1
figure;

% Plotting input signal spectrum
subplot(2, 1, 1);
plot(f_input, X_input_magnitude);
title('Magnitude Spectrum of Original Input Signal');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xlim([-pi pi]);  
xticks(-pi:pi/2:pi); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
grid on;

% Plotting y1 spectrum
subplot(2, 1, 2);
plot(f_y1, Y1_magnitude);  
title('Magnitude Spectrum of Reconstructed Signal y1');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xlim([-pi pi]); 
xticks(-pi:pi/2:pi); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
grid on;


%case 2

op1=filter(H0_0,1,i0); %K0=H0_0
op2=filter(H0_1,1,i1); %K1=H0_1

comm_op1=upsample(op1,2);
comm_op2=upsample(op2,2);
comm_op1=[0;comm_op1]; %We delay the first output by 1
comm_op2=[comm_op2;0]; %We append the second output with a 0 to make the vectors equal in size
y2=comm_op1+comm_op2;%the final output is the sum of these two outputs

%Plotting magnitude spectrum of the original input signal
N_input = length(x);  %Length of the input signal
f_input = linspace(-pi, pi, N_input);  %Frequency vector from -pi to pi for the input signal

X_input = fft(x);  %Computing FFT of input signal
X_input_magnitude = abs(fftshift(X_input));  %Magnitude and shift FFT 

%Plotting magnitude spectrum of the reconstructed signal y2
N_y2 = length(y2); 
f_y2 = linspace(-pi, pi, N_y2); 

Y2 = fft(y2); 
Y2_magnitude = abs(fftshift(Y2));  

% Plotting the magnitude spectrum of both input signal and y2
figure;

% Plotting input signal spectrum
subplot(2, 1, 1);
plot(f_input, X_input_magnitude);
title('Magnitude Spectrum of Original Input Signal');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xlim([-pi pi]);  
xticks(-pi:pi/2:pi); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
grid on;

% Plotting y2 spectrum
subplot(2, 1, 2);
plot(f_y2, Y2_magnitude);  
title('Magnitude Spectrum of Reconstructed Signal y2');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xlim([-pi pi]); 
xticks(-pi:pi/2:pi); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
grid on;

