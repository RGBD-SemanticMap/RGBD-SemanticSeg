% layers = net.layers;
% names = [];
% file = fopen('struct.txt', 'w','native', 'UTF-8');
% for i = 1 : size(layers, 2)
%     names{i} = layers{i}.name;
%     formatSpec = '\t\t|¡ª¡ª %02d. %s\n';
%     fprintf(file, formatSpec, i, names{i})
% end
%% =================================
% layers = net.layers;
% 
% 
% diary on
% for i = 1 : size(layers, 2)
%     
%     disp(['index=' num2str(i)]);
%     layer = layers{i}
%     diary file_pad
%     
% end
% diary off;

%% ============================

diary on
for i = 1 : 85
    
    disp(['================']);
    sz  = size(obj.vars(i).value)


 
    diary multiModel2_varSize
    
end
diary off;