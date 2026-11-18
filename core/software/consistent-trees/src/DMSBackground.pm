package DMSBackground;
use strict;
use warnings;
use POSIX qw(isfinite);

# Same piecewise power-law H(a)/H0 and exact da/(a E) integral as the native
# dms_background.h adapter. H0*(t(a)-t(1)) is negative for past epochs.
my (@a, @e, @time);

sub enabled { return scalar @a; }

sub segment_time {
    my ($a0, $a1, $e0, $e1, $at) = @_;
    my $p = log($e1/$e0)/log($a1/$a0);
    my $l = log($at/$a0);
    my $x = -$p*$l;
    return $l/$e0*(1-$p*$l/2+$p*$p*$l*$l/6) if abs($x)<1e-10;
    my $em1 = abs($x)<1e-5 ? $x*(1+$x/2+$x*$x/6+$x**3/24+$x**4/120) : exp($x)-1;
    return -$em1/($p*$e0);
}

sub init {
    my ($path) = @_;
    @a=(); @e=(); @time=();
    return if !defined($path) || !length($path) || $path eq 'none';
    open my $file, '<', $path or die "Cannot open background $path: $!\n";
    while(my $line=<$file>) {
        next if $line =~ /^\s*(#|$)/;
        my @row = split /\s+/, $line =~ s/^\s+//r;
        die "Invalid background row in $path\n" unless @row>=2 && isfinite($row[0]) && isfinite($row[1])
            && $row[0]>0 && $row[1]>0 && (!@a || $row[0]>$a[-1]);
        push @a, 0+$row[0]; push @e, 0+$row[1];
    }
    close $file;
    die "Background must end at a=1, E=1\n" unless @a>=2 && abs($a[-1]-1)<1e-14 && abs($e[-1]-1)<1e-6;
    $time[0]=0;
    for(my $i=1; $i<@a; $i++) {
        $time[$i]=$time[$i-1]+segment_time($a[$i-1],$a[$i],$e[$i-1],$e[$i],$a[$i]);
    }
}

sub interval_index {
    my ($at)=@_;
    die "Background epoch $at outside supplied range\n" unless isfinite($at) && $at>=$a[0]*(1-1e-14) && $at<=$a[-1]*(1+1e-14);
    my ($low,$high)=(0,$#a);
    while($high-$low>1) {
        my $mid=int(($low+$high)/2);
        if($a[$mid]>$at) {$high=$mid;} else {$low=$mid;}
    }
    return $low;
}

sub E {
    my ($at)=@_;
    my $i=interval_index($at);
    return exp(log($e[$i])+log($at/$a[$i])/log($a[$i+1]/$a[$i])*log($e[$i+1]/$e[$i]));
}

sub time {
    my ($at)=@_;
    my $i=interval_index($at);
    return $time[$i]+segment_time($a[$i],$a[$i+1],$e[$i],$e[$i+1],$at)-$time[-1];
}
1;
