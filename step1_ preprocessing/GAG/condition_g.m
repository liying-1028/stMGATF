
function id = condition_g(adjmc,kk)
a = sum(adjmc,2);
[T,INDEX]=sort(a,'descend');
id = INDEX(1:kk);
end